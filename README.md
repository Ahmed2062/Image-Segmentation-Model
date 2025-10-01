# Instance Segmentation with Mask R-CNN and OpenCV

## Project Goal

The objective of this project is to perform **instance segmentation** on images to detect, classify, and outline individual objects. The system uses a pre-trained Mask R-CNN model loaded via OpenCV's DNN module to identify multiple objects in an image and generate precise pixel-wise masks for each one.

## Approach

The project follows a computer vision pipeline to process an image and render segmentation masks and bounding boxes.

**1. Model Loading**
* The system uses the powerful **Mask R-CNN** architecture, specifically a model pre-trained on the COCO (Common Objects in Context) dataset.
* The pre-trained model files (`frozen_inference_graph_coco.pb` and `mask_rcnn_inception_v2_coco_2018_01_28.pbtxt`) are loaded using OpenCV's `dnn.readNetFromTensorflow` function.
* The corresponding class names from the COCO dataset are loaded from `coco.names`.

**2. Image Preprocessing**
* An input image is loaded and converted into a `blob` using `cv2.dnn.blobFromImage`. A blob is a 4D NumPy array that serves as the standard input format for the neural network. This step also handles necessary preprocessing like resizing and channel swapping.

**3. Model Inference**
* The blob is passed as input to the network.
* A forward pass is performed to obtain two key outputs:
    * `boxes`: The coordinates of the bounding boxes for each detected object.
    * `masks`: The pixel-wise segmentation data for each detected object.

**4. Post-processing and Visualization**
A loop iterates through each detected object to filter out low-confidence detections (score < 0.6) and process the valid ones:
* **Mask Generation:** For each object, the corresponding mask is resized to the dimensions of its bounding box. Thresholding and morphological operations are applied to clean up the mask and create a clear binary outline.
* **Coloring:** A random color is generated for each unique object instance.
* **Contour Detection:** Contours are found for each mask, and `cv2.fillPoly` is used to draw the filled, colored segmentation mask onto a blank image.
* **Final Rendering:** The original image is blended with the colored mask image using `cv2.addWeighted`. This creates a final, visually appealing output where segmented objects are highlighted with a transparent color overlay. Bounding boxes and class labels are also drawn on the final image.

---

## Key Findings and Results

The system successfully performs instance segmentation, accurately identifying and outlining multiple objects within the input image.

**Example Result:**

For the input image of a person on a horse, the model produced the following output:
* **Detected Objects:** `horse` (score: 0.99) and `person` (score: 0.87).
* **Visualization:**
    1.  **Bounding Boxes:** Clear bounding boxes are drawn around both the person and the horse.
    2.  **Segmentation Masks:** Pixel-perfect masks are generated and overlaid with distinct, random colors (e.g., the horse in blue, the person in green).
    3.  **Image Blending:** The final output image clearly shows the original scene with the detected objects highlighted by the transparent masks, making the segmentation easy to interpret.

## Conclusion

This project successfully implements an end-to-end image segmentation pipeline using OpenCV's DNN module and a pre-trained Mask R-CNN model. It demonstrates the ability to not only locate and classify objects but also to understand their exact shape and boundaries at a pixel level. The approach is effective and provides a solid foundation for more advanced computer vision applications like autonomous driving, medical imaging analysis, or robotic interaction. 

## Files used and model architecture

📁 [Click to see the files used for the project](https://drive.google.com/drive/folders/1e7go8lIUhmY3Y6y_B7EbobEo7n_Ox04L?usp=drive_link)

To view the model architecture used in the project you may use the following website for a flow diagram https://netron.app/
