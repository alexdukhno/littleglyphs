import copy
import numpy as np
import scipy
import skimage

# min/max clamps on feature values
feature_clamps_coordinate = (0.1,0.899)
feature_clamps_weight = (0.0,2)

# min/max clamps on feature values
feature_clamps_coordinate = (0.1,0.899)
feature_clamps_weight = (0.0,2)



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
    value_clamps = feature_clamps_coordinate
    value_clamp_low = value_clamps[0]
    value_clamp_high = value_clamps[1]
    
    def __init__(self, values=None):            
        if values != None:
            if len(values) == self.expected_value_count:
                self.values = np.array(values)
                self.N_values = len(self.values)
            else:
                raise(
                    "feature got improper quantity of values: expected "+str(self.expected_value_count)+
                    ", got "+str(len(values))
                )
        else:
            self.values = np.zeros(self.expected_value_count)
            self.N_values = self.expected_value_count
        
    def __repr__(self):
        return 'Feature: type='+str(self.feature_type)+'; values='+str(self.values)    
    
    def set_values(self, values):
        np.copyto(self.values, values)
        self.clamp_values()
    def set_value(self, valueindex, value):
        self.values[valueindex] = value
        self.clamp_values()
    def __len__(self):
        return self.N_values
    
    def clamp_values(self):        
        self.values = np.where(self.values >= self.value_clamp_high, self.value_clamp_high, self.values) 
        self.values = np.where(self.values < self.value_clamp_low, self.value_clamp_low, self.values)
    
    def permute(self, permutation_strength):
        # Permutes the feature in-place.
        self.values = self.values + (
            (np.random.rand(self.N_values)-0.5)*2 * permutation_strength
        )
        self.clamp_values()
        
    def permuted(self, permutation_strength):
        # Returns a permuted copy of the feature.
        new_feature = self.__class__(self.values)
        new_feature.permute(permutation_strength)
        return new_feature
    
    def randomize_value(self, valueindex):
        # Randomizes the feature value in-place.
        self.values[valueindex] = np.random.rand()
        self.clamp_values()   
        
    def randomize_all_values(self):
        # Randomizes ALL feature values in-place.
        self.values = np.random.rand(self.N_values)
        self.clamp_values()        

        
        
class FeatureLineSegment(Feature):
    '''
    A line segment feature.
    Feature should contain 4 values:
    x1,y1,x2,y2 - coordinates of start and endpoints of line segment.
    Image center corresponds to (0.5,0.5).
    '''
    feature_type = 'LineSegment'
    expected_value_count = 4
    
    def render(self, image, mode='set'):
        
        imgsize_x, imgsize_y = np.shape(image)[0], np.shape(image)[1]
        x1,y1,x2,y2 = self.values
        
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

    value_clamps = None
    value_clamp_low = None
    value_clamp_high = None
    N_points = None
    
    def __init__(self, N_points, values=None):            
        if N_points < 3:
            raise('MultiPointLineSegment requires at least three points')
        else:
            self.N_points = N_points
            self.expected_value_count = 4 + (self.N_points-2)*2
            clamps = [feature_clamps_coordinate]*(self.N_points*2)
            self.value_clamps = np.array(clamps)
            self.value_clamp_low = np.array(self.value_clamps[:,0])
            self.value_clamp_high = np.array(self.value_clamps[:,1])
                    
        if values != None:
            if len(values) == self.expected_value_count:
                self.values = np.array(values)
                self.N_values = len(self.values)
            else:
                raise(
                    "feature got improper quantity of values: expected "+str(self.expected_value_count)+
                    ", got "+str(len(values))
                )
        else:
            self.values = np.zeros(self.expected_value_count)
            self.N_values = self.expected_value_count
    
    def render(self, image, mode='set'):        
        imgsize_x, imgsize_y = np.shape(image)[0], np.shape(image)[1]
        
        x1,y1,x2,y2 = self.values[0:4]
        
        x1,x2 = (int(x1*imgsize_x), int(x2*imgsize_x))
        y1,y2 = (int(y1*imgsize_y), int(y2*imgsize_y))
        
        cc,rr = skimage.draw.line(x1,y1,x2,y2)
        if mode=='set':
            image[rr,cc] = 1
        else:
            image[rr,cc] = image[rr,cc] + 1
        
        for i in range(0,self.N_points-2):
            x1, y1 = x2, y2
            x2,y2 = self.values[4+i*2 : 4+(i+1)*2]
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

    value_clamps = np.array([feature_clamps_coordinate]*6 + [feature_clamps_weight])
    value_clamp_low = np.array(value_clamps[:,0])
    value_clamp_high = np.array(value_clamps[:,1])
  
    def render(self, image, mode='set'):
        imgsize_x, imgsize_y = np.shape(image)[0], np.shape(image)[1]
        x1,y1,xc,yc,x2,y2,wc = self.values
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
    x1,y1,xc,tc,x2,y2,wc - coordinates of start, control, and endpoints of the first segment of bezier curve, 
      and weight of control point;
      for every additional point beyond the first two:
    xci,tci,xi,yi,wc - coordinates of control point and endpoint of the next segment, and weight of control point.
      NB: for every next segment the starting point is the endpoint of the previous one.
    image center corresponds to (0.5,0.5).      
    '''
    
    feature_type = 'MultiPointBezierCurve'
    expected_value_count = None

    value_clamps = None
    value_clamp_low = None
    value_clamp_high = None
    N_points = None
    
    def __init__(self, N_points, values=None):            
        if N_points < 3:
            raise('MultiPointBezierCurve requires at least three points')
        else:
            self.N_points = N_points
            self.expected_value_count = 7 + (self.N_points-2)*5
            clamps = [feature_clamps_coordinate]*6 + [feature_clamps_weight] 
            clamps = clamps + ( [feature_clamps_coordinate]*4 + [feature_clamps_weight] )*(self.N_points-2)
            self.value_clamps = np.array(clamps)
            self.value_clamp_low = np.array(self.value_clamps[:,0])
            self.value_clamp_high = np.array(self.value_clamps[:,1])
                    
        if values != None:
            if len(values) == self.expected_value_count:
                self.values = np.array(values)
                self.N_values = len(self.values)
            else:
                raise(
                    "feature got improper quantity of values: expected "+str(self.expected_value_count)+
                    ", got "+str(len(values))
                )
        else:
            self.values = np.zeros(self.expected_value_count)
            self.N_values = self.expected_value_count
    
    def render(self, image, mode='set'):
        
        imgsize_x, imgsize_y = np.shape(image)[0], np.shape(image)[1]
        
        x1,y1,xc,yc,x2,y2,wc = self.values[0:7]
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
        
        for i in range(0,self.N_points-2):
            x1, y1 = x2, y2
            xc,yc,x2,y2,wc = self.values[7+i*5 : 7+(i+1)*5]
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

    def render(self, image, mode='set'):
        
        imgsize_x, imgsize_y = np.shape(image)[0], np.shape(image)[1]
        x1,y1,x2,y2 = self.values
        x1,x2 = (int(x1*imgsize_x), int(x2*imgsize_x))
        y1,y2 = (int(y1*imgsize_y), int(y2*imgsize_y))
        xc,yc = ((x1+x2)//2, (y1+y2)//2)
        rx,ry = (abs((x1-x2)//2), abs((y1-y2)//2))

        cc,rr = skimage.draw.ellipse_perimeter(xc,yc,rx,ry, shape = (imgsize_x,imgsize_y))
        if mode=='set':
            image[rr,cc] = 1
        else:
            image[rr,cc] = image[rr,cc] + 1
            
