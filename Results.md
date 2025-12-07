# Results – Blocks World Vision System

---

## Overview
The preprocessing pipeline was applied to images `image1.png` through `image9.png`.  
The pipeline performs grayscale conversion, smoothing, binary segmentation, edge detection, edge classification (vertical/horizontal), figure-ground segmentation, and contact/occlusion edge detection.  

All preprocessing results, including edge maps, masks, and 3D reconstruction visualizations, are displayed in the **Block World Vision Model.ipynb** notebook.

---

## Preprocessing Summary

| Image       | Total Edges | Vertical Edges | Horizontal Edges | Contact Edges | Occlusion Edges |
|------------|-------------|----------------|-----------------|---------------|----------------|
| image1.png | 9670        | 874            | 8796            | 2             | 52             |
| image2.png | 9990        | 3247           | 6743            | 3             | 217            |
| image3.png | 12111       | 5578           | 6533            | 2             | 107            |
| image4.png | 28514       | 11097          | 17417           | 8             | 433            |
| image5.png | 12761       | 74             | 12687           | 2             | 46             |
| image6.png | 15656       | 7091           | 8565            | 36            | 211            |
| image7.png | 8160        | 2782           | 5378            | 1             | 112            |
| image8.png | 31250       | 8085           | 23165           | 139           | 481            |
| image9.png | 6657        | 1188           | 5469            | 1             | 94             |

---

## Observations

- Edge detection successfully identifies block boundaries in all images.  
- Vertical and horizontal edges are classified using gradient orientation.  
- Contact edges (where blocks touch the ground) are minimal but correctly detected.  
- Occlusion edges (overlaps between blocks) are detected effectively, especially in complex scenes.  
- Total edge counts vary with image complexity and number of blocks.  

---

## Visualization

All results including:  
- Original images  
- Binary segmentation  
- Edge maps  
- Edge classification (vertical/horizontal)  
- Contact and occlusion edges  
- 3D reconstruction surfaces  

are available and can be interactively explored in **Block World Vision Model.ipynb**.

---
