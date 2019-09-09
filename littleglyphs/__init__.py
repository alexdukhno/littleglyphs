import copy
import numpy as np
import scipy
import skimage

feature_clamps_coordinate = (0.1,0.899)
feature_clamps_weight = (0.01,2)

class Feature:    
    # a Feature denotes a graphical primitive (e.g. curve, circle, 
    #   straight line segment, bezier curve, etc.).
    # 'featuretype' denotes which kind of a primitive the feature is.
    # 'values' denote which parameters are to be used to invoke 
    #    this feature when drawing a Glyph (see below).
    # 'value_clamps' denote the permitted lower and upper bounds on
    #    values.
    
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
    # A line segment feature.
    # Feature should contain 4 values:
    # x1,y1,x2,y2 - coordinates of start and endpoints of line segment.
    # Image center corresponds to (0.5,0.5).
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
    # A composite (multipoint) line segment.
    # Feature should contain 4+2*(N_points-2) values:
    # x1,y1,x2,y2 - coordinates of start and endpoints of the first segment;    
    #   for every additional point beyond the first two:
    # xi,yi - coordinates of control point and endpoint of the next segment.
    #   NB: for every next segment the starting point is the endpoint of the previous one.
    # image center corresponds to (0.5,0.5).
      
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
    # A simple bezier feature: two endpoints, one control point and its weight.
    # Feature should contain 7 values:
    # x1,y1,xc,tc,x2,y2,wc - coordinates of start, control, and endpoints of bezier curve, and weight of control point.
    # Image center corresponds to (0.5,0.5).
        
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

        cc,rr = skimage.draw.bezier_curve(x1,y1,xc,yc,x2,y2,wc, shape = (imgsize_x,imgsize_y))
        if mode=='set':
            image[rr,cc] = 1
        else:
            image[rr,cc] = image[rr,cc] + 1

            
            
class FeatureMultiPointBezierCurve(Feature):
    # A composite (multipoint) bezier feature.
    # Feature should contain 7+5*(N_points-2) values:
    # x1,y1,xc,tc,x2,y2,wc - coordinates of start, control, and endpoints of the first segment of bezier curve, 
    #   and weight of control point;
    #   for every additional point beyond the first two:
    # xci,tci,xi,yi,wc - coordinates of control point and endpoint of the next segment, and weight of control point.
    #   NB: for every next segment the starting point is the endpoint of the previous one.
    # image center corresponds to (0.5,0.5).
      
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

            cc,rr = skimage.draw.bezier_curve(x1,y1,xc,yc,x2,y2,wc, shape = (imgsize_x,imgsize_y))
            if mode=='set':
                image[rr,cc] = 1
            else:
                image[rr,cc] = image[rr,cc] + 1
               
            
            
            
            
class FeatureEllipse(Feature):
    # An ellipse feature.
    # Feature should contain 4 values:
    # x1,y1,x2,y2 - coordinates of top-left and bottom-right points of bounding rectangle
    # as floats with values from 0 to 1; image center corresponds to (0.5,0.5).

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
            
        
# --- Glyph related ---    
                
        
class Glyph:
    # a Glyph represents a list of graphical primitives (e.g. curves, circles, 
    #   straight line segments, bezier curves, etc.), which, when drawn in their order, 
    #   construct the glyph. Think vector graphics.
    # A glyph is constructed in the order in which its elements appear in the list.
    #   For instance, for a glyph comprised of [circle, square, triangle],
    #   if there is any overlap between the circle, square, and triangle,
    #   then the topmost would be triangle and bottommost would be circle.
    # A glyph has a semantical category to which it belongs. This is useful when you want several glyphs
    #   to be semantically identical (e.g. a set of glyphs for 'zero' could be a circle, 
    #   an ellipse or a crossed-out ellipse).
    # A semantical category should be of an integer type.
    
    def __init__(self, features, category=None):
        self.features = copy.deepcopy(features)
        self.category = copy.deepcopy(category)
        self.N_features = len(self.features)
    def add_feature(self, feature):
        self.features.append(feature)
        self.N_features = len(self.features)
    def remove_feature(self, feature_index):
        del self.features[feature_index]        
        self.N_features = len(self.features)
    
    def set_category(self, category):
        self.category = category
    
    def permute(self,permutation_strength):
        # Permutes glyph features in-place.
        for feature in self.features:
            feature.permute(permutation_strength)

    def permuted(self,permutation_strength):
        # Returns a permuted copy of the glyph.
        new_glyph = self.__class__(self.features, self.category)
        new_glyph.permute(permutation_strength)
        return new_glyph
            
    def randomize_all_features(self):
        # Randomizes ALL features in-place.
        for feature in self.features:
            feature.randomize_all_values()
            
    def render(
        self, imgsize, 
        blur=True,
        blur_factor=0.5,
        up_down_sample=True,
        normalize=True,
        mode='set'
    ):
        # Renders the glyph to a raster, given imgsize (tuple of (imgsize_x, imgsize_y)).    
        # blur_factor determines how much the image should be blurred before output.
        # up_down_sample determines if the raster should be constructed from an upsampled image
        #   and then downsampled (can be important for operations that set individual pixels).
        # normalize determines if the raster should be normalized by dividing by its maximum.
        
        if isinstance(imgsize,tuple):        
            imgsize_x, imgsize_y = imgsize[0], imgsize[1]
        else:
            imgsize_x = imgsize
            imgsize_y = imgsize
        
        if up_down_sample:
            imgsize_x, imgsize_y = imgsize_x*2, imgsize_y*2

        
        image = np.zeros((imgsize_x,imgsize_y))
        
        for feature in self.features:
            feature.render(image, mode=mode)
        
        if blur:
            image = scipy.ndimage.filters.gaussian_filter(image, sigma=blur_factor)
            
        if up_down_sample:
            image_rescaled = skimage.transform.rescale(image, 0.5, anti_aliasing=True, multichannel=False)
            image = image_rescaled
        
        if normalize:
            img_max = np.amax(image)
            image = image / img_max
        
        return image
            

    
    
class GlyphList:
    # a GlyphList is a list of glyphs.
    
    def __init__(self, glyphs):
        self.glyphs = glyphs
        self.N_glyphs = len(self.glyphs)
    
    def __len__(self):
        return self.N_glyphs
    def add_glyph(self, glyph):
        self.glyphs.append(glyph)
        self.N_glyphs = len(self.glyphs)
    def remove_glyph(self, glyph_index):
        del self.glyphs[glyph_index]        
        self.N_glyphs = len(self.glyphs)    
    
    def remove_glyph_category(self, removed_category):
        indices_to_remove = []
        for i, glyph in enumerate(self.glyphs):
            if glyph.category == removed_category:
                indices_to_remove.append(i)
        self.glyphs = [glyph for i, glyph in enumerate(self.glyphs) if i not in indices_to_remove]        
        self.N_glyphs = len(self.glyphs)
                
    def permuted(self, permutation_strength, N_glyph_permutations, 
                 keep_original_glyphs=False):
        # Makes a new copy of the glyphlist.
        # Each new glyph is a permuted copy of the old one.
        # Categories are preserved for permuted glyphs.
        # If keep_original_glyphs is True, include the original glyphs at the beginning of the list.
        
        if keep_original_glyphs:
            new_glyphs = self.glyphs
        else:
            new_glyphs = []
            
        for glyph in self.glyphs:
            for i in range(0, N_glyph_permutations):
                new_glyph = glyph.permuted(permutation_strength)
                new_glyphs.append(new_glyph)
        
        new_glyphlist = self.__class__(new_glyphs)
        return new_glyphlist
    
    def render(
        self, imgsize, 
        blur=True,
        blur_factor=0.5,
        randomize_blur=False,
        random_blur_extent=2,
        up_down_sample=True,
        normalize=True,
        mode='set'
    ):
        if isinstance(imgsize,tuple):            
            rasters = np.zeros((self.N_glyphs,*(imgsize)))
        else:
            rasters = np.zeros((self.N_glyphs,imgsize,imgsize))
        
        categories = []
        for i, glyph in enumerate(self.glyphs):
            if randomize_blur:
                blur_factor_for_current_raster = blur_factor * np.power(random_blur_extent, (np.random.rand()-0.5)*2)
            else:
                blur_factor_for_current_raster = blur_factor
            rasters[i] = glyph.render(
                imgsize, 
                blur, blur_factor_for_current_raster,
                up_down_sample, normalize,
                mode=mode
            )
            
            categories.append(glyph.category)
            
        raster_array = RasterArray(rasters, categories)
        return raster_array

    
    
#--- Raster related ---

# A 'raster' is a float rectangular grayscale raster image, with values in [0,1] range.




class RasterArray:
    # a RasterArray is a wrapper for a numpy array containing a stack of rasters:
    #   rectangular grayscale matrices of identical size, with float values in [0,1] range.
    # 'N_rasters' contains the quantity of rasters in RasterArray.
    #   The actual data is stored in 'rasters' variable, 
    #   but it can be also accessed by direct indexing of RasterArray.
    # First dimension refers to raster index in the array.
    # 'categories' is a helper numpy array containing the categories that the rasters pertain to.
    
    def __init__(self, rasters, categories):
        self.rasters = np.array(rasters)  
        self.categories = np.array(categories)
        self.N_rasters = self.rasters.shape[0]
        self.imgsize = self.rasters.shape[1]
        # future: extend to rectangular images
        self.imgsize_x = self.rasters.shape[1]
        self.imgsize_y = self.rasters.shape[2]
        
    def __getitem__(self, key):
        return self.rasters[key]
    
    def __len__(self):
        return self.N_rasters
    
    def repeated(self,N_times):
        # Concatenates the raster array N_times in a row, keeping track of categories.
        new_rasters = self.rasters.repeat(N_times,axis=0)
        new_categories = self.categories.repeat(N_times,axis=0)
        new_raster_array = self.__class__(new_rasters,new_categories)
        return new_raster_array
    
    def distort(self, distorter):
        # Distorts the RasterArray in-place by the passed distorter.
        for i in range(self.N_rasters):
            self.rasters[i] = distorter.apply(self.rasters[i])
        
    def distorted(self, distorter, N_distortions_per_raster):
        # Generates a new RasterArray containing images distorted by the passed distorter.
        # N_distortions_per_image controls how many new distortions are made.
        if N_distortions_per_raster > 0:
            new_raster_array = self.repeated(N_distortions_per_raster)
        else:
            raise('Cannot generate fewer than one distortion')
        new_raster_array.distort(distorter)
        return new_raster_array
    

#--- Distortion related ---
    
    
    
class Distortion:
    # A distortion is a wrapper for a function that distorts a raster image, 
    #   using a certain set of parameters.
    distortion_type = None
    def __init__(self, **kwargs):
        self.params = kwargs
        


class DistortionRandomAffine(Distortion):
    # Distorts an image with a random affine transformation.
    # As the more extreme deformations may potentially lead "useful" information in the original image 
    #   outside of the output image limits, the image is padded beforehand and then cropped around the 
    #   image centroid to recenter the image.
    # 
    # Parameters for transformation are chosen uniformly randomly from: 
    #   -rotat_distort_max to rotat_distort_max, 
    #   -shear_distort_max to shear_distort_max, 
    #   -scale_distort_max to scale_distort_max.
        
    distortion_type = 'RandomAffine'
    
    def __init__(self, **kwargs):
        self.params = kwargs
        self.rotat_distort_max = self.params['rotat_distort_max']
        self.shear_distort_max = self.params['shear_distort_max']
        self.scale_distort_max = self.params['scale_distort_max']
        
    def apply(self, image):
        imgsize = max(image.shape[0],image.shape[1])
        newimg = np.copy(image)
        
        transform = skimage.transform.AffineTransform(
            scale= ((1 - ((np.random.rand()-0.5)*self.scale_distort_max)*2),
                    (1 - ((np.random.rand()-0.5)*self.scale_distort_max)*2)
                   ), 
            rotation= (np.random.rand()-0.5)*2*self.rotat_distort_max, 
            shear= (np.random.rand()-0.5)*2*self.shear_distort_max,
            translation=None            
        )    
        newimg = skimage.util.pad(newimg,pad_width=imgsize*3, mode='constant')
        newimg = skimage.transform.warp(newimg,transform)
        newimg = skimage.util.pad(newimg,pad_width=imgsize*3, mode='constant')
        newimg_moments = skimage.measure.moments(newimg,order=1)
        centroid_x = int(newimg_moments[1,0]/newimg_moments[0,0])
        centroid_y = int(newimg_moments[0,1]/newimg_moments[0,0])
        distorted_image = np.copy(
            newimg[
                centroid_x-imgsize//2:centroid_x+imgsize//2, 
                centroid_y-imgsize//2:centroid_y+imgsize//2
            ]            
        )
        return distorted_image
    

    
class RandomSequentialDistorter:
    # a RandomSequentialDistorter is a sequence of distortion functions to apply to an image.
    # Takes a list of elements equal to [Distortion instance, probability] 
    #   and sequentially applies the distortions to the image.
    #   Each distortion is applied with probability from [0, 1].
    
    
    def __init__(self, distortion_and_probability_list):
        self.distortions = [item[0] for item in distortion_and_probability_list]
        self.probabilities = [item[1] for item in distortion_and_probability_list]
    
    def apply(self, image):
        newimage = np.copy(image)
        for i in range(0, len(self.distortions)):
            rnum = np.random.rand()
            if rnum < self.probabilities[i]:
                newimage = self.distortions[i].apply(newimage)
        return newimage
    
    
    
class SequentialDistorter(RandomSequentialDistorter):
    # A sequential distorter is a RandomsSequentialDistorter with all probabilities set to 1.
    def __init__(self, distortions):
        self.distortions = distortions
        self.probabilities = [1 for item in distortions]
    
