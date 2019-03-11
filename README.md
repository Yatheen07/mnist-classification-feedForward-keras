# mnist-classification-feedForward-keras
All the blogs has explained to implement the feed forward networks, but checking the model for our own input is missing in many sites.

## Steps to implement the model for own input is discussed here.

### Step 1: Create a basic image of a digit using paint.

### Step 2: Resize the image to 28 x 28 pixels. [As i have used it as a metric provided by the dataset]

        (https://user-images.githubusercontent.com/26171078/41290618-0eafbfa0-6e6b-11e8-9a02-317805033619.png)

### Step 3: Use openCV to read the image as a vector.
         
         import cv2
         image = cv2.imread(r'Your file path here')

### Step 4: Convert it into an numpy array.
         
         test_image = np.array(image) 
         
### Step 5: Linearise the array values into a single-valued channel
        
        dim = np.prod(test_image.shape[:])
        result_image = test_image.reshape(1, dim)
        
### Step 6: Convert the pixel values so that they fall in the range of 0 to 1

        result_image = result_image.astype('float32')
        result_image /= 255
        
### Step 7: Use the model to predict the result

        model_reg.predict_classes(result_image)

