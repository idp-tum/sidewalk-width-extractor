import cv2
from argparse import ArgumentParser, Namespace
import numpy as np
from skimage.morphology import skeletonize

def get_args() -> Namespace:
    """Parse given arguments

    Returns:
        Namespace: parsed arguments
    """

    parser = ArgumentParser()

    parser.add_argument(
        "--source_mask",
        type=str,
        required=True,
        help= "Feed extracted sidewalks to get their average width",
    )

    return parser.parse_args()

def load_mask(
    mask_path: str
):
    """Loads mask from given path

    Returns:
        Return mask in greyscale (numpy.ndarray)
    """

    img = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

    return img

def get_area(mask)->int:

    return (mask == 255).sum()    


def skeletonize(img):
    
    size = np.size(img)
    skel = np.zeros(img.shape,np.uint8)
    
    ret,img = cv2.threshold(img,127,255,0)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    done = False
    
    while( not done):
        eroded = cv2.erode(img,element)
        temp = cv2.dilate(eroded,element)
        temp = cv2.subtract(img,temp)
        skel = cv2.bitwise_or(skel,temp)
        img = eroded.copy()
    
        zeros = size - cv2.countNonZero(img)
        if zeros==size:
            done = True
    
    #cv2.imshow("skel",skel)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    return skel


def get_width(
    mask_path: str
) -> int:

    """Compute average sidewalk width

    Returns:
        Sidewalk width in metres.
    """

    mask = load_mask(mask_path)

    area = get_area(mask)

    skeletonized_mask = skeletonize(mask)

    sum_skeleton = (skeletonized_mask == 255).sum()
    
    return area/sum_skeleton * 20 / 100



if __name__ == "__main__":
    args = get_args()
    
    print(get_width(args.source_mask))
    
    