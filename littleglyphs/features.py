import copy
import numpy as np
import scipy
import skimage
import functools
import bisect

# --- Utilities and parameters for feature clamping ---

def round_to_closest_element(a,x):
    if bisect.bisect(a,x) == len(a):
        res = a[len(a)-1]
    elif bisect.bisect(a,x) == 0:
        res = a[0]
    else:
        l = a[bisect.bisect(a,x)-1]
        r = a[bisect.bisect(a,x)]
        if abs(l - x) < abs(r - x):
            res = l
        else:
            res = r
    return res

# min/max clamps on feature values
value_types = [
    'coordinate',
    'bezier_control_point_weight'
]
value_clamps = {
    'coordinate': (0.1,0.899), 
    'bezier_control_point_weight': (0.0,2.0)
}

feature_clamps_coordinate = (0.1,0.899)
feature_clamps_weight = (0.0, 2.0)

# grid dimensions for blocky feature clamping
blockgrid_coordinate = 3
blockgrid_weight = 4
value_blockgrids = {
    'coordinate': np.linspace(*feature_clamps_coordinate,blockgrid_coordinate),
    'bezier_control_point_weight': np.linspace(*feature_clamps_weight,blockgrid_weight)
}



# --- Decorators ---

def value_clamper(func):
    @functools.wraps(func)
    def wrapper_decorator(self, *args, **kwargs):        
        value = func(self, *args, **kwargs)      
        self.clamp_values()
        return value
    return wrapper_decorator

# --- Classes ---

class Value():
    value_name = None
    value = None
    value_type = None
    def __init__(self, value_name=None, value=None, value_type=None):
        self.value_name = copy.copy(value_name)
        self.value = copy.copy(value)
        self.value_type = copy.copy(value_type)

        
        
class Values():
    values = []
    values_dict = {}
    def __init__(self, values=None):
        self.values = values
        for i in range(0,len(values)):
            self.values_dict[self.values[i].value_name] = i
    def __getitem__(self, key):
        if isinstance(key, str):
            return self.values[self.values_dict[key]]
        else:
            return self.values[key]
    def __setitem__(self, key, value):
        if isinstance(key, str):
            self.values[self.values_dict[key]].value = value
        else:
            self.values[key].value = value
        
        
    
class Feature:    
    '''
    a Feature denotes a graphical primitive (e.g. curve, circle, 
      straight line segment, bezier curve, etc.).
    'featuretype' denotes which kind of a primitive the feature is.
    'values' denote which parameters are to be used to invoke 
       this feature when drawing a Glyph (see below).
    'value_clamps' denote the permitted lower and upper bounds on
       values.
    '''
    
    feature_type = None    
    expected_value_count = 0    
    blocky = False
    
    def __init__(self, values=None, blocky=False):            
        if values != None:
            if len(values) == self.expected_value_count:
                self.values = values
                self.N_values = len(self.values)
            else:
                raise ('feature got improper quantity of values: expected '+str(self.expected_value_count)+
                    ', got '+str(len(values)))
        else:
            self.values = Values(
                [Value(value_name=str(i), value=0, value_type='coordinate') 
                 for i in range(0,self.expected_value_count)]
            )
            self.N_values = self.expected_value_count
        self.blocky = blocky
        
    def __repr__(self):
        return 'Feature: type='+str(self.feature_type)+'; values='+str(self.values)    

    def __len__(self):
        return self.N_values    
    
    def clamp_values(self):        
        for v in self.values:
            if v.value < value_clamps[v.value_type][0]:
                v.value = value_clamps[v.value_type][0]
            elif v.value > value_clamps[v.value_type][1]:
                v.value = value_clamps[v.value_type][1]
            if self.blocky:
                v.value = round_to_closest_element(value_blockgrids[v.value_type], v.value)
                    
    @value_clamper
    def set_values(self, values):
        self.values = copy.deepcopy(values)

    @value_clamper        
    def set_value(self, valueindex, value):
        self.values[valueindex] = copy.deepcopy(value)

    @value_clamper        
    def permute(self, permutation_strength):
        # Permutes the feature in-place.
        for v in self.values:
            v.value = v.value + ((np.random.random()-0.5) * 2 * permutation_strength)
        
    def permuted(self, permutation_strength):
        # Returns a permuted copy of the feature.
        new_feature = self.__class__(self.values)
        new_feature.permute(permutation_strength)
        return new_feature
    
    @value_clamper
    def randomize_value(self, valueindex):
        # Randomizes the feature value in-place.
        self.values[valueindex].value = np.random.random()
    
    @value_clamper
    def randomize_all_values(self):
        # Randomizes ALL feature values in-place.
        for v in self.values:
            v.value = np.random.random()
        
        
        
class FeatureLineSegment(Feature):
    '''
    A line segment feature.
    Feature should contain 4 values:
    x1,y1,x2,y2 - coordinates of start and endpoints of line segment.
    Image center corresponds to (0.5,0.5).
    '''
    feature_type = 'LineSegment'
    expected_value_count = 4

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)        
        self.values[0].value_type = 'coordinate'; self.values[0].value_name = 'x1'
        self.values[1].value_type = 'coordinate'; self.values[1].value_name = 'y1'
        self.values[2].value_type = 'coordinate'; self.values[2].value_name = 'x2'
        self.values[3].value_type = 'coordinate'; self.values[3].value_name = 'y2'
        self.values = Values(self.values.values)
    
    def render(self, image, mode='set'):
        
        imgsize_x, imgsize_y = np.shape(image)[0], np.shape(image)[1]
        x1,y1,x2,y2 = [self.values['x1'].value, self.values['y1'].value, 
                       self.values['x2'].value, self.values['y2'].value]
        
        x1,x2 = (int(x1*imgsize_x), int(x2*imgsize_x))
        y1,y2 = (int(y1*imgsize_y), int(y2*imgsize_y))
        
        cc,rr = skimage.draw.line(x1,y1,x2,y2)
        if mode=='set':
            image[rr,cc] = 1
        else:
            image[rr,cc] = image[rr,cc] + 1

        
                
class FeatureMultiPointLineSegment(Feature):
    '''
    A composite (multipoint) line segment.
    Feature should contain 4+2*(N_points-2) values:
    x1,y1,x2,y2 - coordinates of start and endpoints of the first segment;    
      for every additional point beyond the first two:
    xi,yi - coordinates of control point and endpoint of the next segment.
      NB: for every next segment the starting point is the endpoint of the previous one.
    image center corresponds to (0.5,0.5).
    '''
        
    feature_type = 'MultiPointLineSegment'
    expected_value_count = None
    N_points = None
    
    
    def __init__(self, N_points, *args, **kwargs):            
        if N_points < 3:
            raise ('MultiPointLineSegment requires at least three points')
        else:
            self.N_points = N_points
            self.expected_value_count = 4 + (self.N_points-2)*2

        super().__init__(*args, **kwargs)
        
        for i in range(0,self.expected_value_count,2):
            self.values[i].value_type = 'coordinate'; self.values[i].value_name = 'x'+str(i//2+1)
            self.values[i+1].value_type = 'coordinate'; self.values[i+1].value_name = 'y'+str(i//2+1)   
        
        self.values = Values(self.values.values)
        
    
    def render(self, image, mode='set'):        
        imgsize_x, imgsize_y = np.shape(image)[0], np.shape(image)[1]
        
        x1,y1,x2,y2 = [self.values['x1'].value, self.values['y1'].value, 
                       self.values['x2'].value, self.values['y2'].value]
        
        x1,x2 = (int(x1*imgsize_x), int(x2*imgsize_x))
        y1,y2 = (int(y1*imgsize_y), int(y2*imgsize_y))
        
        cc,rr = skimage.draw.line(x1,y1,x2,y2)
        if mode=='set':
            image[rr,cc] = 1
        else:
            image[rr,cc] = image[rr,cc] + 1
        
        for i in range(2,self.N_points):
            x1, y1 = x2, y2
            x2,y2 = [ self.values['x'+str(i+1)].value, self.values['y'+str(i+1)].value]
            x2,y2 = (int(x2*imgsize_x), int(y2*imgsize_y))
            
            cc,rr = skimage.draw.line(x1,y1,x2,y2)
            if mode=='set':
                image[rr,cc] = 1
            else:
                image[rr,cc] = image[rr,cc] + 1
        
        
        
        
class FeatureBezierCurve(Feature):
    '''
    A simple bezier feature: two endpoints, one control point and its weight.
    Feature should contain 7 values:
    x1,y1,xc,tc,x2,y2,wc - coordinates of start, control, and endpoints of bezier curve, and weight of control point.
    Image center corresponds to (0.5,0.5).
    '''
    
    feature_type = 'BezierCurve'
    expected_value_count = 7

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)        
        self.values[0].value_type = 'coordinate'; self.values[0].value_name = 'x1'
        self.values[1].value_type = 'coordinate'; self.values[1].value_name = 'y1'
        self.values[2].value_type = 'coordinate'; self.values[2].value_name = 'xc1'
        self.values[3].value_type = 'coordinate'; self.values[3].value_name = 'yc1'
        self.values[4].value_type = 'coordinate'; self.values[0].value_name = 'x2'
        self.values[5].value_type = 'coordinate'; self.values[1].value_name = 'y2'
        self.values[6].value_type = 'bezier_control_point_weight'; self.values[0].value_name = 'wc1'
        self.values = Values(self.values.values)
  
    def render(self, image, mode='set'):
        imgsize_x, imgsize_y = np.shape(image)[0], np.shape(image)[1]
        x1,y1,xc,yc = [self.values['x1'].value, self.values['y1'].value, 
                       self.values['xc1'].value, self.values['yc1'].value]
        x2,y2,wc = [self.values['x2'].value, self.values['y2'].value, self.values['wc1'].value]
        x1,xc,x2 = (int(x1*imgsize_x), int(xc*imgsize_x), int(x2*imgsize_x))
        y1,yc,y2 = (int(y1*imgsize_y), int(yc*imgsize_y), int(y2*imgsize_y))
                
        # skimage has an issue with drawing bezier curves:
        #   if the weight for a bezier curve is lower than ca. 0.3 (and not equal to 0),
        #   it will occasionally fail to draw about a half of the curve.
        # Following constant is used for clipping the value when drawing:
        #   if the desired value of weight is 0.00-0.15, it will be set to 0,
        #   otherwise if it's 0.15-0.3, it will be set to 0.3, 
        #   otherwise the value is preserved.
        feature_clamps_weight_min_for_drawing = 0.3 
        
        if wc < (feature_clamps_weight_min_for_drawing + feature_clamps_weight[0])/2:
            wc = 0
        elif wc < (feature_clamps_weight_min_for_drawing):
            wc = feature_clamps_weight_min_for_drawing
        else: 
            wc = wc
            
        cc,rr = skimage.draw.bezier_curve(x1,y1,xc,yc,x2,y2,wc, shape = (imgsize_x,imgsize_y))
        if mode=='set':
            image[rr,cc] = 1
        else:
            image[rr,cc] = image[rr,cc] + 1

            
            
class FeatureMultiPointBezierCurve(Feature):
    '''
    A composite (multipoint) bezier feature.
    Feature should contain 7+5*(N_points-2) values:
    x1,y1,xc,yc,x2,y2,wc - coordinates of start, control, and endpoints of the first segment of bezier curve, 
      and weight of control point;
      for every additional point beyond the first two:
    xci,tci,xi,yi,wc - coordinates of control point and endpoint of the next segment, and weight of control point.
      NB: for every next segment the starting point is the endpoint of the previous one.
    Image center corresponds to (0.5,0.5).      
    '''
    
    feature_type = 'MultiPointBezierCurve'
    expected_value_count = None
    N_points = None
    
    def __init__(self, N_points, *args, **kwargs):            
        if N_points < 3:
            raise ('MultiPointBezierCurve requires at least three points')
        else:
            self.N_points = N_points
            self.expected_value_count = 7 + 5*(self.N_points-2)

        super().__init__(*args, **kwargs)
        
        self.values[0].value_type = 'coordinate'; self.values[0].value_name = 'x1'
        self.values[1].value_type = 'coordinate'; self.values[1].value_name = 'y1'        
        for i in range(2,self.expected_value_count,5):
            self.values[i].value_type = 'coordinate'; self.values[i].value_name = 'xc'+str(i//5+1)
            self.values[i+1].value_type = 'coordinate'; self.values[i+1].value_name = 'yc'+str(i//5+1)
            self.values[i+2].value_type = 'coordinate'; self.values[i+2].value_name = 'x'+str(i//5+2)
            self.values[i+3].value_type = 'coordinate'; self.values[i+3].value_name = 'y'+str(i//5+2)   
            self.values[i+4].value_type = 'bezier_control_point_weight'; self.values[i+4].value_name = 'wc'+str(i//5+2)
        
        self.values = Values(self.values.values)
    
    
    def render(self, image, mode='set'):
        
        imgsize_x, imgsize_y = np.shape(image)[0], np.shape(image)[1]
        
        x1,y1,xc,yc = [self.values['x1'].value, self.values['y1'].value, 
                       self.values['xc1'].value, self.values['yc1'].value]
        x2,y2,wc = [self.values['x2'].value, self.values['y2'].value, self.values['wc1'].value]
        x1,xc,x2 = (int(x1*imgsize_x), int(xc*imgsize_x), int(x2*imgsize_x))
        y1,yc,y2 = (int(y1*imgsize_y), int(yc*imgsize_y), int(y2*imgsize_y))
        
        # skimage has an issue with drawing bezier curves:
        #   if the weight for a bezier curve is lower than ca. 0.3 (and not equal to 0),
        #   it will occasionally fail to draw about a half of the curve.
        # Following constant is used for clipping the value when drawing:
        #   if the desired value of weight is 0.00-0.15, it will be set to 0,
        #   otherwise if it's 0.15-0.3, it will be set to 0.3, 
        #   otherwise the value is preserved.
        feature_clamps_weight_min_for_drawing = 0.3 
        
        if wc < (feature_clamps_weight_min_for_drawing + feature_clamps_weight[0])/2:
            wc = 0
        elif wc < (feature_clamps_weight_min_for_drawing):
            wc = feature_clamps_weight_min_for_drawing
        else: 
            wc = wc
        
        cc,rr = skimage.draw.bezier_curve(x1,y1,xc,yc,x2,y2,wc, shape = (imgsize_x,imgsize_y))
        if mode=='set':
            image[rr,cc] = 1
        else:
            image[rr,cc] = image[rr,cc] + 1
        
        for i in range(2,self.N_points):
            x1, y1 = x2, y2
            x2,y2,xc,yc,wc = [
                self.values['x'+str(i+1)].value, self.values['y'+str(i+1)].value, 
                self.values['xc'+str(i)].value, self.values['yc'+str(i)].value, 
                self.values['wc'+str(i)].value
            ]
            xc,x2 = (int(xc*imgsize_x), int(x2*imgsize_x))
            yc,y2 = (int(yc*imgsize_y), int(y2*imgsize_y))
            
            if wc < (feature_clamps_weight_min_for_drawing + feature_clamps_weight[0])/2:
                wc = 0
            elif wc < (feature_clamps_weight_min_for_drawing):
                wc = feature_clamps_weight_min_for_drawing
            else: 
                wc = wc

            cc,rr = skimage.draw.bezier_curve(x1,y1,xc,yc,x2,y2,wc, shape = (imgsize_x,imgsize_y))
            if mode=='set':
                image[rr,cc] = 1
            else:
                image[rr,cc] = image[rr,cc] + 1
               
            
            
            
            
class FeatureEllipse(Feature):
    '''
    An ellipse feature.
    Feature should contain 4 values:
    x1,y1,x2,y2 - coordinates of top-left and bottom-right points of bounding rectangle
    as floats with values from 0 to 1; image center corresponds to (0.5,0.5).
    '''
    
    feature_type = 'Ellipse'
    expected_value_count = 4

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)        
        self.values[0].value_type = 'coordinate'; self.values[0].value_name = 'x1'
        self.values[1].value_type = 'coordinate'; self.values[1].value_name = 'y1'
        self.values[2].value_type = 'coordinate'; self.values[2].value_name = 'x2'
        self.values[3].value_type = 'coordinate'; self.values[3].value_name = 'y2'
        self.values = Values(self.values.values)    
    
    def render(self, image, mode='set'):        
        imgsize_x, imgsize_y = np.shape(image)[0], np.shape(image)[1]
        x1,y1,x2,y2 = [self.values['x1'].value, self.values['y1'].value, 
                       self.values['x2'].value, self.values['y2'].value]
        x1,x2 = (int(x1*imgsize_x), int(x2*imgsize_x))
        y1,y2 = (int(y1*imgsize_y), int(y2*imgsize_y))
        xc,yc = ((x1+x2)//2, (y1+y2)//2)
        rx,ry = (abs((x1-x2)//2), abs((y1-y2)//2))

        cc,rr = skimage.draw.ellipse_perimeter(xc,yc,rx,ry, shape = (imgsize_x,imgsize_y))
        if mode=='set':
            image[rr,cc] = 1
        else:
            image[rr,cc] = image[rr,cc] + 1
            
