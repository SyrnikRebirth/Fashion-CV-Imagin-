from cropper import Cropper
from skimage.exposure import is_low_contrast
import numpy as np
import math
import PIL


class Face_quality:
    def __init__(self, 
                 contrast_param=0.225, 
                 face_size_param=200, 
                 face_in_pic_ratio=0.3, 
                 eyes_to_face_param = 0.35
        ):
        self.cr = Cropper()
        self.contrast_param = contrast_param
        self.face_size_param = face_size_param
        self.face_in_pic_ratio = face_in_pic_ratio
        self.eyes_to_face_param = eyes_to_face_param

    def check_face_quality(self, image):
        widht = np.shape(image)[1]
        hight = np.shape(image)[0]
        
        landmarks = cr.get_mp_landmarks(image)
        if landmarks is None:
            raise Exception("I can't find face")
        
        face_cords = self.cr.get_face_rectangle(image)
        try:
            picture_is_contrast_enogh(image)
            face_is_big_enogh(face_cords)
            face_fully_in_picture(widht, hight, face_cords)
            eyes_to_face_ratio(image, landmarks)
        except Exception as e:
            print(e)
            
            
    def picture_is_contrast_enogh(self, image):
        #We use skimage function, that measures contrast
        if is_low_contrast(image, self.contrast_param):
            raise Exception("This picture is too dim") 
        return True
    
    def face_is_big_enogh(self, face_cords):
        #We check whether there are enough pixels for a normal face reckognition
        if face_cords[2] <= self.face_size_param:
            raise Exception("Face size is smaller than " + str(self.face_size_param) + " pixels") 
        return True
    
        
    def face_fully_in_picture(self, image_widht, image_hight, face_cords):
        picture_square = image_widht * image_hight
        
        # points that frame face
        point1 = (face_cords[0], face_cords[1])
        point2 = (face_cords[0] + face_cords[2], face_cords[1])
        point3 = (face_cords[0], face_cords[1] + face_cords[2])
        point4 = (face_cords[0] + face_cords[2], face_cords[1] + face_cords[2])
        
        # checking if this points are out of borders of image
        face_points = [point1, point2, point3, point4]
        list_of_unfitted_pixels_cords = []

        for point in face_points:
            if point[0] >= image_widht or point[0] <= 0:
                list_of_unfitted_pixels_cords.append(point)
            if point[1] >= image_hight or point[1] <= 0:
                list_of_unfitted_pixels_cords.append(point)
                
        
        # function for calculating the square        
        def two_cords_square(p0, p1):
            x = 0
            y = 0
            if p0[0] == p1[0]:
                x = abs(widht - p0[0])
                y = abs(p0[1] - p1[1])
            elif p0[1] == p1[1]:
                x = abs(p0[0] - p1[0])
                y = abs(image_hight - p0[1])
            else:
                x = abs(p0[0] - p1[0])
                y = abs(p0[1] - p1[1])
            return x*y

        # processing all possible variants
        diff = 0.
        if len(list_of_unfitted_pixels_cords) == 2:
            p0 = list_of_unfitted_pixels_cords[0]
            p1 = list_of_unfitted_pixels_cords[1]
            diff = two_cords_square(p0, p1)

        if len(list_of_unfitted_pixels_cords) == 3:
            p0 = list_of_unfitted_pixels_cords[0]
            p1 = list_of_unfitted_pixels_cords[1]
            p2 = list_of_unfitted_pixels_cords[1]
            diff += two_cords_square(p0, p1)
            diff += two_cords_square(p1, p2)
    
            twice_counted = (abs(p1[0] - image_widht), abs(p1[1] - image_hight))
            diff -= twice_counted[0] * twice_counted[1]
    
        if len(list_of_unfitted_pixels_cords) == 4:
            p0 = list_of_unfitted_pixels_cords[0]
            p1 = list_of_unfitted_pixels_cords[-1]
            diff = two_cords_square(p0, p1) - picture_square
        
        # finally check if face is in picture enough
        if diff / picture_square > self.face_in_pic_ratio:
            raise Exception("More than " + str(self.face_in_pic_ratio * 100) + "% of your face is not in the picture") 
            
        return True
    
    
    def eyes_to_face_ratio(self, image, landmarks):
        # firstly, we will rotate picture, so eyes are at the same level
        
        # get eyes centers in pixels
        r_center = cr.get_center_of_right_eye(landmarks)
        l_center = cr.get_center_of_left_eye(landmarks)

        pixel_r_center = (round(r_center[0] * widht), round(r_center[1] * hight))
        pixel_l_center = (round(l_center[0] * widht), round(l_center[1] * hight))

        # assume direction, of rotation 
        direction = 1
        if pixel_r_center[1] < pixel_l_center[1]:
            direction = -1
    
        # get info that we need, to calculate angle
        x_pixel_distance = abs(pixel_r_center[0] - pixel_l_center[0])
        y_pixel_distance = abs(pixel_r_center[1] - pixel_l_center[1])
        curr_distance_between_eyes = round(np.sqrt(x_pixel_distance ** 2 + y_pixel_distance ** 2))

        # calculate angle
        angle_cos = x_pixel_distance / curr_distance_between_eyes
        angle = np.arccos(angle_cos)

        # turn it from radians to degrees
        angle = (angle * 180) / math.pi

        # rotate image
        new_img = PIL.Image.fromarray(image)
        new_img = np.array(new_img.rotate(direction * angle))

        # find face cords and landmarks of new picture
        new_face_cords = cr.get_face_rectangle(new_img)
        new_landmarks = cr.get_mp_landmarks(new_img)
        new_r_center = cr.get_center_of_right_eye(new_landmarks)
        new_l_center = cr.get_center_of_left_eye(new_landmarks)
        
        # calculate ratio between face rectangle side and distance between eyes
        final_distance_between_eyes = abs(round(new_r_center[0] * widht) - round(new_l_center[0] * widht))
        eyes_to_face_ratio = final_distance_between_eyes / new_face_cords[2] 
        
        if eyes_to_face_ratio > 0.35:
            raise Exception("It seems that your face is turned too much to the side, think about taking another picture") 
        return True
