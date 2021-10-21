from fed.iou import compute_iou
from fed import config as cfg
from bs4 import BeautifulSoup
from imutils import paths
import cv2
import os

for dirpath in (cfg.GEN_FIRE_PATH, cfg.GEN_NON_FIRE_PATH):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

imagePaths = list(paths.list_images(cfg.DETECT_IMAGES))

totalFires = 0
totalNonFires = 0

for (i, imagePath) in enumerate(imagePaths):
    print("[INFO] processing image {}/{}...".format(i + 1, len(imagePaths)))
    filename = imagePath.split(os.path.sep)[-1]
    filename = filename[:filename.rfind(".")]
    annotPath = os.path.sep.join(
        [cfg.DETECT_ANNOTS, "{}.xml".format(filename)])

    contents = open(annotPath).read()
    soup = BeautifulSoup(contents, "html.parser")
    gtBoxes = []

    w = int(soup.find("width").string)
    h = int(soup.find("height").string)

    for o in soup.find_all("object"):
        # extract the label and bounding box coordinates
        label = o.find("name").string
        xMin = int(o.find("xmin").string)
        yMin = int(o.find("ymin").string)
        xMax = int(o.find("xmax").string)
        yMax = int(o.find("ymax").string)

        # truncate any bounding box coordinates that may fall
        # outside the boundaries of the image
        xMin = max(0, xMin)
        yMin = max(0, yMin)
        xMax = min(w, xMax)
        yMax = min(h, yMax)

        # update our list of ground-truth bounding boxes
        gtBoxes.append((xMin, yMin, xMax, yMax))

    image = cv2.imread(imagePath)

    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(image)
    ss.switchToSelectiveSearchFast()
    boxes = ss.process()
    proposedBoxes = []

    for (x, y, w, h) in boxes:
        proposedBoxes.append((x, y, x + w, y + h))

    fireROIs = 0
    nonFireROIs = 0

    for proposedBox in proposedBoxes[:cfg.MAX_PROPOSALS]:
        (ppStartX, ppStartY, ppEndX, ppEndY) = proposedBox

        for gtBox in gtBoxes:
            iou = compute_iou(gtBox, proposedBox)
            (gtStartX, gtStartY, gtEndX, gtEndY) = gtBox

            roi = None
            outputPath = None

            if iou > 0.7 and fireROIs <= cfg.MAX_FIRE:
                roi = image[ppStartY:ppEndY, ppStartX:ppEndX]
                filename = "{}.jpg".format(totalFires)
                outputPath = os.path.sep.join(
                    [cfg.GEN_FIRE_PATH, filename])

                fireROIs += 1
                totalFires += 1

            fullOverlap = ppStartX >= gtStartX
            fullOverlap = fullOverlap and ppStartY >= gtStartY
            fullOverlap = fullOverlap and ppEndX <= gtEndX
            fullOverlap = fullOverlap and ppEndY <= gtEndY

            if not fullOverlap and iou < 0.05 and nonFireROIs <= cfg.MAX_NON_FIRE:
                roi = image[ppStartX:ppEndY, ppStartX:ppEndX]
                filename = "{}.jpg".format(totalNonFires)
                outputPath = os.path.sep.join(
                    [cfg.GEN_NON_FIRE_PATH, filename])

                nonFireROIs += 1
                totalNonFires += 1

            if roi is not None and outputPath is not None:
                try:
                    roi = cv2.resize(roi, cfg.INPUT_DIMS,
                                     interpolation=cv2.INTER_CUBIC)
                    cv2.imwrite(outputPath, roi)
                except Exception as e:
                    print("[WARNING] skipped roi")
