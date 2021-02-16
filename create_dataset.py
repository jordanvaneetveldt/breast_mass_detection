import numpy as np
import cv2 
import glob
import pydicom as dicom
import random
from pathlib import Path
from sklearn.model_selection import train_test_split
import shutil
import os
import albumentations as A
import plistlib
from skimage.draw import polygon

DCM_PATH = 'INbreast Release 1.0/AllDICOMs/'
XML_PATH = 'INbreast Release 1.0/AllXML/'

MASS_PATIENT_ID = ['53586896', '22580192', '22614236', '22580098', '24055445', '30011674', '20586934', '22670465', '24055502', '22670673', '20587612', '22614568', '20587902', '22614522', '50995789', '24055464', '20588216', '51049053', '53582656', '20588562', '27829188', '22614431', '22580341', '22613822', '24065584', '50997515', '51049107', '22580367', '22580244', '50996352', '22670147', '22580732', '50999008', '24065707', '22614127', '20588334', '20588536', '24065530', '22670324', '20586908', '30011507', '27829134', '53581406', '50998981', '20586986', '22678787', '50997461', '53580804', '22579730', '22670094', '53580858', '53586869', '50995762', '24065251', '20587810', '53581460', '22670855', '22580706', '30011553', '22670809', '22580419', '24055355', '53587014', '50994408', '22614379', '22670278', '24065289', '22614074', '24055274', '22670511', '50994354', '20587928', '22580393', '22580654', '20588046', '50994273', '20587758', '24065761', '22427751', '20587664', '50999432', '22580680', '22580038', '53587663', '20588308', '20588680', '30011727', '22678833', '22427705', '22614266', '22613650', '50999459', '24055483', '22678694', '20587994', '22678646', '53582683', '20586960', '51048765', '22670620', '22613770', '22427840', '20588190', '53586960', '50996406', '22613702', '51048738']

seed= 40 #to generate a different dataset

# flip, shift, rotate augmentation
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.ShiftScaleRotate(scale_limit = 0, rotate_limit=180, p=1, border_mode=0)
], bbox_params=A.BboxParams(format='yolo' , min_visibility=0.4))


def load_inbreast_mask(mask_path, imshape=(4084, 3328)):
    """
    This function loads a osirix xml region as a binary numpy array for INBREAST
    dataset
    @mask_path : Path to the xml file
    @imshape : The shape of the image as an array e.g. [4084, 3328]
    return: numpy array where each mass has a different number id.
    """

    def load_point(point_string):
        x, y = tuple([float(num) for num in point_string.strip('()').split(',')])
        return y, x
    i =  0
    mask = np.zeros(imshape)
    with open(mask_path, 'rb') as mask_file:
        plist_dict = plistlib.load(mask_file, fmt=plistlib.FMT_XML)['Images'][0]
        numRois = plist_dict['NumberOfROIs']
        rois = plist_dict['ROIs']
        assert len(rois) == numRois
        for roi in rois:
            numPoints = roi['NumberOfPoints']
            if roi['Name'] == 'Mass':
                i+=1
                points = roi['Point_px']
                assert numPoints == len(points)
                points = [load_point(point) for point in points]
                if len(points) <= 2:
                    for point in points:
                        mask[int(point[0]), int(point[1])] = i
                else:
                    x, y = zip(*points)
                    x, y = np.array(x), np.array(y)
                    poly_x, poly_y = polygon(x, y, shape=imshape)
                    mask[poly_x, poly_y] = i
    return mask

def mask_to_yolo(mass_mask):
    """
    Convert a mask into albumentations format. 
    @mass_mask : numpy array mask where each pixel correspond to a lesion (one pixel id per lesion)
    return: a list of list containing masses bounding boxes in YOLO coordinates:
            <x> = <absolute_x> / <image_width>
            <y> = <absolute_y> / <image_height>
            <height> = <absolute_height> / <image_height>
            <width> = <absolute_width> / <image_width>
    """
    res = []
    height, width = mass_mask.shape
    nbr_mass = len(np.unique(mass_mask))-1
    
    for i in range(nbr_mass):
        mask = mass_mask.copy()
        mask[mass_mask!=i+1]=0
        #find contours of each mass
        cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #create a bbox around the contours
        x, y, w, h = cv2.boundingRect(cnts[0])
        #convert to yolo coordinate system
        x = x+w//2 -1
        y= y+h//2 -1
        res.append([x/width,y/height,w/width,h/height, 'mass'])
    return res

def bbox_to_txt(bboxes):
    """
    Convert a list of bbox into a string in YOLO format (to write a file).
    @bboxes : numpy array of bounding boxes 
    return : a string for each object in new line: <object-class> <x> <y> <width> <height>
    """
    txt=''
    for l in bboxes:
        l = [str(x) for x in l[:4]]
        l = ' '.join(l)
        txt += '0 ' + l + '\n'
    return txt

def crop(img, mask):
    """
    Crop breast ROI from image.
    @img : numpy array image
    @mask : numpy array mask of the lesions
    return: numpy array of the ROI extracted for the image, 
            numpy array of the ROI extracted for the breast mask,
            numpy array of the ROI extracted for the masses mask
    """
    # Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(img,(5,5),0)
    _, breast_mask = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    cnts, _ = cv2.findContours(breast_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(cnts, key = cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)

    return img[y:y+h, x:x+w], breast_mask[y:y+h, x:x+w], mask[y:y+h, x:x+w]

def truncation_normalization(img, mask):
    """
    Clip and normalize pixels in the breast ROI.
    @img : numpy array image
    @mask : numpy array mask of the breast
    return: numpy array of the normalized image
    """
    Pmin = np.percentile(img[mask!=0], 5)
    Pmax = np.percentile(img[mask!=0], 99)
    truncated = np.clip(img,Pmin, Pmax)  
    normalized = (truncated - Pmin)/(Pmax - Pmin)
    normalized[mask==0]=0
    return normalized

def clahe(img, clip):
    """
    Image enhancement.
    @img : numpy array image
    @clip : float, clip limit for CLAHE algorithm
    return: numpy array of the enhanced image
    """
    clahe = cv2.createCLAHE(clipLimit=clip)
    cl = clahe.apply(np.array(img*255, dtype=np.uint8))
    return cl

def synthetized_images(patient_id):
    """
    Create a 3-channel image composed of the truncated and normalized image,
    the contrast enhanced image with clip limit 1, 
    and the contrast enhanced image with clip limit 2 
    @patient_id : patient id to recover image and mask in the dataset
    return: numpy array of the breast region, numpy array of the synthetized images, numpy array of the masses mask
    """
    image_path = glob.glob(os.path.join(DCM_PATH,str(patient_id)+'*.dcm'))[0]
    mass_mask = load_inbreast_mask(os.path.join(XML_PATH,str(patient_id)+'.xml'))
    ds = dicom.dcmread(image_path)
    pixel_array_numpy = ds.pixel_array

    breast, mask, mass_mask = crop(pixel_array_numpy, mass_mask)
    normalized = truncation_normalization(breast, mask)

    cl1 = clahe(normalized, 1.0)
    cl2 = clahe(normalized, 2.0)

    synthetized = cv2.merge((np.array(normalized*255, dtype=np.uint8),cl1,cl2))
    return breast, synthetized, mass_mask


if __name__ == "__main__":
    train_set, test_set = train_test_split(MASS_PATIENT_ID, test_size = 0.1, random_state=seed)
    train_set, val_set = train_test_split(train_set, test_size = 0.11, random_state=seed)

    shutil.rmtree('data/', ignore_errors = True) 
    Path('data/obj').mkdir(parents=True, exist_ok=True)
    Path('data/test').mkdir(parents=True, exist_ok=True) #test is val in yolo
    Path('data/unseen').mkdir(parents=True, exist_ok=True)

    cntr = 0

    for patient_id in train_set:
        original, synthetized, mass_mask = synthetized_images(patient_id)
        width = 800
        height = min(int(synthetized.shape[0] * 800 / synthetized.shape[1]), 1333)
        dim = (width, height) 


        synthetized = cv2.resize(synthetized, dim, interpolation = cv2.INTER_AREA) 
        mass_mask = cv2.resize(mass_mask, dim, interpolation = cv2.INTER_NEAREST) 

        bboxes = mask_to_yolo(mass_mask)
  
        for i in range(8):
            transformed = transform(image=synthetized, bboxes=bboxes)
            transformed_image = transformed['image']
            transformed_bboxes = transformed['bboxes']
            if transformed_bboxes != []:
                txt = bbox_to_txt(transformed_bboxes)
                cv2.imwrite(os.path.join('data/obj/', '%d.png'%cntr), transformed_image)

                txt_file = open(os.path.join('data/obj/', '%d.txt'%cntr), "w")
                txt_file.write(txt)
                txt_file.close()
                cntr+=1

    for patient_id in val_set:
        original, synthetized, mass_mask = synthetized_images(patient_id)

        width = 800
        height = min(int(synthetized.shape[0] * 800 / synthetized.shape[1]), 1333)
        dim = (width, height) 

        synthetized = cv2.resize(synthetized, dim, interpolation = cv2.INTER_AREA) 
        mass_mask = cv2.resize(mass_mask, dim, interpolation = cv2.INTER_NEAREST) 
        txt = bbox_to_txt(mask_to_yolo(mass_mask))
        cv2.imwrite(os.path.join('data/test/', '%d.png'%cntr), synthetized)

        txt_file = open(os.path.join('data/test/', '%d.txt'%cntr), "w")
        txt_file.write(txt)
        txt_file.close()
        cntr+=1

    #same code as for val set
    for patient_id in test_set:
        original, synthetized, mass_mask = synthetized_images(patient_id)

        width = 800
        height = min(int(synthetized.shape[0] * 800 / synthetized.shape[1]), 1333)
        dim = (width, height) 

        synthetized = cv2.resize(synthetized, dim, interpolation = cv2.INTER_AREA) 
        mass_mask = cv2.resize(mass_mask, dim, interpolation = cv2.INTER_NEAREST) 
        txt = bbox_to_txt(mask_to_yolo(mass_mask))
        cv2.imwrite(os.path.join('data/unseen/', '%d.png'%cntr), synthetized)

        txt_file = open(os.path.join('data/unseen/', '%d.txt'%cntr), "w")
        txt_file.write(txt)
        txt_file.close()
        cntr+=1

# labelImg
classes_txt = open('data/classes.txt', "w")
classes_txt.write('mass')
classes_txt.close()