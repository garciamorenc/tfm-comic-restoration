from augraphy import *
from augraphy.default.pipeline import *
import cv2 as cv2

pipeline = default_augraphy_pipeline()
pipeline = pipeline_archetype10()
ink_phase = [DirtyDrum()]
paper_phase = [ColorPaper()]
post_phase = [BleedThrough()]

pipeline = AugraphyPipeline(ink_phase, paper_phase, post_phase)

img = cv2.imread("inputs/comic1.jpg")

image_augmented = pipeline(img)


# display image
cv2.namedWindow('test', cv2.WINDOW_NORMAL)
cv2.imshow("test", image_augmented)
cv2.waitKey()
cv2.destroyAllWindows()