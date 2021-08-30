from header_imports import *


class edge_detection_analysis(object):
    def __init__(self, input_auguments):

        # Import paths
        print("here")
        self.input_auguments = input_auguments[1]
        self.image_path = self.input_auguments + "/"
        self.images = [count for count in glob(self.image_path +'*') if 'jpg' in count]


        # Personal Kernel Creation 
        self.kernel_2 = kernel2 = np.array([[1, -3, -1],
                                            [3, 0, 3],
                                            [1, -3, -1]]) 
        

        # Original edge detection kernel
        self.edge_detector = np.array([[-1, -1, -1],
                                        [-1, 8, -1],
                                        [-1, -1, -1]])
    

        # Count from the standard
        self.count = 0
        
        # Start analysing
        self.image_looping()
        print("here")
        


    def save_image(self, img, file_name, image_to_save):

        
        if image_to_save == 'train_edge_1':
            image_output = "train_edge_1/"
        elif image_to_save == 'test_edge_1':
            image_output = "test_edge_1"
        elif image_to_save == "val_edge_1":
            image_output = "val_edge_1"
        elif image_to_save == 'train_edge_2':
            image_output = "train_edge_2/"
        elif image_to_save == 'test_edge_2':
            image_output = "test_edge_2"
        elif image_to_save == "val_edge_2":
            image_output = "val_edge_2"


        
        for i in range(len(self.images)):
            cv2.imwrite(os.path.join(image_output, str(file_name)), img)
            cv2.waitKey(0)
        

    
    def image_looping(self):
        
        for image in self.images:
            self.count +=1
            print(self.count)

            img = cv2.imread(image, -1)
            file_name = basename(image)
    
            gray_scale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            img_my_kernel_2 = cv2.filter2D(gray_scale, -1, self.kernel_2)
            edge_detector_known = cv2.filter2D(gray_scale, -1, self.edge_detector)

            self.save_image(img_my_kernel_2, file_name, image_to_save = self.input_auguments + "_edge_1")
            self.save_image(edge_detector_known, file_name, image_to_save = self.input_auguments + "_edge_2")
