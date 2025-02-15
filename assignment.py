# -*- coding: utf-8 -*-
"""\
------------------------------------------------------------
PanoramaUI class used to implement the UI, feature detection,
feature matching and image stitching.

Made by Jose Casimiro Revez
------------------------------------------------------------\
"""

import cv2
import numpy as np 
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import filedialog

class PanoramaUI:
    
    def __init__(self, root):
        
        self.root = root
        self.root.title("Image Stitching UI")
        
        # Default values for feature detection and matching
        self.ratio_test = False
        self.feature_detection = "sift"
        self.des_type = "sift"
        
        
        # Creating the frame to display the images
        self.image_frame = tk.Frame(root)
        self.image_frame.pack(side=tk.TOP, padx=10, pady=10)

        # Initialising the image display labels
        self.image_label1 = tk.Label(self.image_frame)
        self.image_label2 = tk.Label(self.image_frame)
        self.image_label1.grid(row=0, column=0, padx=5, pady=5)
        self.image_label2.grid(row=0, column=1, padx=5, pady=5)

        # Creating the buttons frame
        self.button_frame = tk.Frame(root)
        self.button_frame.pack(side=tk.TOP, padx=10, pady=10)

        # Creating the buttons
        self.load_button = tk.Button(self.button_frame, text="Load Images", command=self.load_images)
        self.detect_button = tk.Button(self.button_frame, text="Display Detected Features", command=self.display_detected_features)
        self.matches_button = tk.Button(self.button_frame, text="Display Matches", command=self.display_matches)
        self.stitch_button = tk.Button(self.button_frame, text="Stitch Images", command=self.stitch_images)
        
        # Placing the buttons on the UI
        self.load_button.grid(row=0, column=0, padx=5)
        self.detect_button.grid(row=0, column=1, padx=5)
        self.matches_button.grid(row=0, column=2, padx=5)
        self.stitch_button.grid(row=0, column=3, padx=5)
        
        ## Checkbox to select whether the ratio test is used ##
        
        self.cb_ratio_var = tk.IntVar(value=0)
        self.cb_ratio = tk.Checkbutton(self.button_frame, text="Use Ratio Test", variable=self.cb_ratio_var, command=self.update_variables)
        self.cb_ratio.grid(row=1, column=0, padx=5)
        
        ## Radio Button to select either sift or harris corner for displaying the features ##
        
        self.radio_label1 = tk.Label(self.button_frame, text="Select the feature detection method.")
        self.radio_var1 = tk.IntVar()

        self.radio1_sift = tk.Radiobutton(self.button_frame, text="SIFT Feature Point Detection", variable=self.radio_var1, value=1, command=self.update_variables)
        self.radio1_harris = tk.Radiobutton(self.button_frame, text="Harris Corner Detection", variable=self.radio_var1, value=2, command=self.update_variables)

        self.radio_label1.grid(row=2, column=0, padx=5)
        self.radio1_sift.grid(row=2, column=1, padx=5)
        self.radio1_harris.grid(row=2, column=2, padx=5)
        
        ## Radio button to chose if sift or orb descriptors are used ##
        
        self.radio_label2 = tk.Label(self.button_frame, text="Select which kind of descriptors to use.")
        self.radio_var2 = tk.IntVar()

        self.radio2_sift = tk.Radiobutton(self.button_frame, text="SIFT", variable=self.radio_var2, value=1, command=self.update_variables)
        self.radio2_orb = tk.Radiobutton(self.button_frame, text="ORB", variable=self.radio_var2, value=2, command=self.update_variables)

        self.radio_label2.grid(row=3, column=0, padx=5)
        self.radio2_sift.grid(row=3, column=1, padx=5)
        self.radio2_orb.grid(row=3, column=2, padx=5)
    
    ## Function to update the variables selected
    # using the radio buttons and checkbox
    def update_variables(self):
        
        cb_ratio_var = self.cb_ratio_var.get()
        radio_var1 = self.radio_var1.get()
        radio_var2 = self.radio_var2.get()
        
        if cb_ratio_var == 0:
            self.ratio_test = False
        else:
            self.ratio_test = True
            
        if radio_var1 == 1:
            self.feature_detection = "sift"
        else:
            self.feature_detection = "harris"
            
        if radio_var2 == 1:
            self.des_type = "sift"
        else:
            self.des_type = "orb"
            
    ## Function handling 
    def display_images(self):
        
        # Converting images to RGB
        image1_rgb = cv2.cvtColor(self.image1, cv2.COLOR_BGR2RGB)
        image2_rgb = cv2.cvtColor(self.image2, cv2.COLOR_BGR2RGB)

        # Resizing the images
        image1_resized = cv2.resize(image1_rgb, (450, 300))
        image2_resized = cv2.resize(image2_rgb, (450, 300))

        # Converting images to PIL format
        image1_pil = Image.fromarray(image1_resized)
        image2_pil = Image.fromarray(image2_resized)

        # Converting PIL images to Tkinter PhotoImage
        photo1 = ImageTk.PhotoImage(image1_pil)
        photo2 = ImageTk.PhotoImage(image2_pil)

        # Updating image labels
        self.image_label1.configure(image=photo1)
        self.image_label2.configure(image=photo2)
        self.image_label1.image = photo1
        self.image_label2.image = photo2
    
    ## Function to create a new window and display the results
    def display_results(self, num_images, image_list, window_title):
        if num_images == 1:
            
            # Resizing the image
            image_resized = cv2.resize(image_list[0], (1200, 400))

            # Converting images to PIL format
            image_pil = Image.fromarray(image_resized)

            # Converting PIL images to Tkinter PhotoImage
            photo = ImageTk.PhotoImage(image_pil)
            
            # Create a new window to display the stitched image
            result_window = tk.Toplevel(self.root)
            result_window.title(window_title)

            # Create labels to display the images
            img_label = tk.Label(result_window, image=photo)
            img_label.pack()
            
            # Keep a reference to the PhotoImage to prevent it from being garbage collected
            img_label.image = photo
        elif num_images == 2:
            
            # Resizing the images
            image1_resized = cv2.resize(image_list[0], (600, 400))
            image2_resized = cv2.resize(image_list[1], (600, 400))

            # Converting images to PIL format
            image1_pil = Image.fromarray(image1_resized)
            image2_pil = Image.fromarray(image2_resized)

            # Converting PIL images to Tkinter PhotoImage
            photo1 = ImageTk.PhotoImage(image1_pil)
            photo2 = ImageTk.PhotoImage(image2_pil)
            
            # Create a new window to display the stitched image
            result_window = tk.Toplevel(self.root)
            result_window.title(window_title)

            # Create labels to display the images
            img1_label = tk.Label(result_window, image=photo1)
            img2_label = tk.Label(result_window, image=photo2)
            img1_label.pack(side=tk.LEFT)
            img2_label.pack(side=tk.RIGHT)
            
            # Keep a reference to the PhotoImage to prevent it from being garbage collected
            img1_label.image = photo1
            img2_label.image = photo2
            
    # Harris corner detection implemented using opencv functions
    def harris_corner(self):
        
        image1 = cv2.cvtColor(self.image1, cv2.COLOR_BGR2GRAY)
        image2 = cv2.cvtColor(self.image2, cv2.COLOR_BGR2GRAY)
        
        img1_with_cp = cv2.cvtColor(self.image1, cv2.COLOR_BGR2RGB)
        img2_with_cp = cv2.cvtColor(self.image2, cv2.COLOR_BGR2RGB)
        
        # Performing Harris corner detection
        harris_image1 = cv2.cornerHarris(image1, 2, 3, 0.04)
        harris_image2 = cv2.cornerHarris(image2, 2, 3, 0.04)
        
        dst1 = cv2.dilate(harris_image1,None)
        dst2 = cv2.dilate(harris_image2,None)
        
        # Thresholding the corner response to obtain the corner points
        threshold1 = 0.01 * dst1.max()
        threshold2 = 0.01 * dst2.max()
        
        # Marking the points on the image as blue dots
        img1_with_cp[dst1>threshold1]=[0,0,255]
        img2_with_cp[dst2>threshold2]=[0,0,255]
        
        return img1_with_cp, img2_with_cp
    
    # SIFT feature point detection implemented using opencv functions
    def sift(self, img1, img2):
        
        image1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        image2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # Create SIFT object
        sift = cv2.SIFT_create()
        
        # Detect keypoints in the images
        keyp1, des1 = sift.detectAndCompute(image1, None)
        keyp2, des2 = sift.detectAndCompute(image2, None)
        
        return keyp1, keyp2, des1, des2
    
    # Function to obtain ORB descriptors using opencv functions
    def orb(self, img1, img2):
        
        image1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        image2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # Create SIFT object
        orb = cv2.ORB_create()
        
        # Detect keypoints in the images
        keyp1, des1 = orb.detectAndCompute(image1, None)
        keyp2, des2 = orb.detectAndCompute(image2, None)
        
        return keyp1, keyp2, des1, des2
    
    # Function returning the matches between two images.
    # This is worked out using the BFMatcher class on opencv
    def match_features(self, keyp1, keyp2, des1, des2):
        
        # create BFMatcher object using NORM_L2 as distance measure (SSD)
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
         
        # Using ratio test if it is selected on the UI
        if self.ratio_test == True:
            
            # Getting the two best matches for every feature
            matches = bf.knnMatch(des1,des2,k=2)
 
            # Applying ratio test
            ratio_matches = []
            for m,n in matches:
                # ratio of 0.5 means the best distance needs to be at least
                # twice the second best distance to be considered a good match
                if m.distance < 0.5*n.distance:
                    ratio_matches.append(m)
            matches = ratio_matches
            
        else:
            # Getting the matches
            matches = bf.match(des1,des2)
            # Sorting them by distance
            matches = sorted(matches, key = lambda x:x.distance)
        
        return matches
    
    # Function to open the file explorer so the two images can be selected
    def load_images(self):
        file_paths = filedialog.askopenfilenames()
        if len(file_paths) == 2:
            self.image1 = cv2.imread(file_paths[0])
            self.image2 = cv2.imread(file_paths[1])
            self.display_images()
    
    # Function that checks which feature detection method is selected in the UI
    # and performs feature detection, displaying the results in a new window
    def display_detected_features(self):
        if self.feature_detection == "sift":
            
            # Converting images to RGB
            image1_rgb = cv2.cvtColor(self.image1, cv2.COLOR_BGR2RGB)
            image2_rgb = cv2.cvtColor(self.image2, cv2.COLOR_BGR2RGB)
            
            # Perform SIFT feature detection
            keyp1, keyp2, des1, des2 = self.sift(self.image1, self.image2)
            
            image1_with_keyps = cv2.drawKeypoints(image1_rgb,keyp1,image1_rgb)
            image2_with_keyps = cv2.drawKeypoints(image1_rgb,keyp1,image2_rgb)
            
            self.display_results(2, [image1_with_keyps,image2_with_keyps], "Features Detected Using the SIFT Feature Point Detection Method")
            
        elif self.feature_detection == "harris":
            
            # Perform Harris corner feature detection
            img1_with_cp, img2_with_cp = self.harris_corner()
            
            self.display_results(2, [img1_with_cp,img2_with_cp], "Features Detected Using the Harris Corner Detection Method")

    # Function to check which types of descriptors are selected in the UI
    # and performs feature matching, displaying the results in a new window
    def display_matches(self):
        
        # Converting images to RGB
        image1_rgb = cv2.cvtColor(self.image1, cv2.COLOR_BGR2RGB)
        image2_rgb = cv2.cvtColor(self.image2, cv2.COLOR_BGR2RGB)
        
        if self.des_type=="sift":
            keyp1, keyp2, des1, des2 = self.sift(self.image1, self.image2)
            matches = self.match_features(keyp1, keyp2, des1, des2)
            # Draw first 350 matches.
            matched_image = cv2.drawMatches(image1_rgb,keyp1,image2_rgb,keyp2,matches[:350],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            self.display_results(1, [matched_image], "Matched Features Using SIFT Descriptors")
        elif self.des_type == "orb":
            keyp1, keyp2, des1, des2 = self.orb(self.image1, self.image2)
            matches = self.match_features(keyp1, keyp2, des1, des2)
            # Draw first 350 matches.
            matched_image = cv2.drawMatches(image1_rgb,keyp1,image2_rgb,keyp2,matches[:350],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            self.display_results(1, [matched_image], "Matched Features Using BRIEF Descriptors")
            
    # Function performing SIFT feature detection, finding the matches
    # between the two images, warping the second image, stitching the images
    # together and finally displaying the final result in a new window
    def stitch_images(self):
        
        # Converting images to RGB
        image1_rgb = cv2.cvtColor(self.image1, cv2.COLOR_BGR2RGB)
        image2_rgb = cv2.cvtColor(self.image2, cv2.COLOR_BGR2RGB)
        
        keyp1, keyp2, des1, des2 = self.sift(self.image2, self.image1)
        matches = self.match_features(keyp1, keyp2, des1, des2)
        src_pts = np.float32([ keyp1[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
        dst_pts = np.float32([ keyp2[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)
         
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        
        # Warp image2 to align with image1
        height1, width1 = image1_rgb.shape[:2]
        height2, width2 = image2_rgb.shape[:2]
        panorama = cv2.warpPerspective(image2_rgb, M, (width1 + width2, height1))
        
        # Place image1 in the panorama
        panorama[0:height1, 0:width1] = image1_rgb
        
        self.display_results(1, [panorama], "Stitched Images")

if __name__ == "__main__":
    root = tk.Tk()
    app = PanoramaUI(root)
    root.mainloop()