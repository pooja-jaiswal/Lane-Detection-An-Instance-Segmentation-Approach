import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

ext = ".jpg"
path_root = "generated/testing_data/"
#path_root = "generated/training_data/"
path_image = path_root + "image/"
path_binary = path_root + "image_binary/"
path_instance = path_root + "image_instance/"

str_img = "drive/apps/data/testing_data/image/" #0000.png 
str_bin = "drive/apps/data/testing_data/image_binary/" #0000.png 
str_ins = "drive/apps/data/testing_data/image_instance/" #0000.png
#str_img = "drive/apps/data/training_data/image/" #0000.png 
#str_bin = "drive/apps/data/training_data/image_binary/" #0000.png 
#str_ins = "drive/apps/data/training_data/image_instance/" #0000.png

outF = open("val.txt", "w")
#outF = open("train.txt", "w")

file_name_counter = 0
jsonFiles = ["test_label.json"]
for fileCounter in range(len(jsonFiles)):
    json_gt = [json.loads(line) for line in open(str(jsonFiles[fileCounter]))]

    # Looping over the single file from upper loop line by line
    for cnt in range(len(json_gt)): #range(1,10):
        gt = json_gt[cnt]
        gt_lanes = gt['lanes']
        y_samples = gt['h_samples']
        raw_file = gt['raw_file']

        #print(raw_file)

        #img_test = plt.imread("10.jpg")
        ######## Saving Original Image File ##########
        path_image_updated = path_image + str(file_name_counter) + ext
        img = plt.imread(raw_file)
        plt.imsave(path_image_updated,img)
        print(img.shape)
        #plt.imshow(img)
        #plt.show()

        '''
            'raw_file': str. Clip file path.
            'lanes': list. A list of lanes. For each list of one lane, the elements are width values on image.
            'h_samples': list. A list of height values corresponding to the 'lanes', which means len(h_samples) == len(lanes[i])
        '''
        # Here For each lanes loop fetched its x from lanes[i] and 
        # y from h_samples which is same for all lane for that particular JSON ROW
        #gt_lanes_vis = [[(x, y) for (x, y) in zip(lane, y_samples) if x >= 0] for lane in gt_lanes]
        
        #List of Lanes are samerated from the JSON file 
        # that can be passed to the polylines for plotting
        lane_list = []
        for lane in gt_lanes:
            single_lane = []
            for (x,y) in zip(lane, y_samples):
                if(x > 0):
                    single_lane.append([x,y])
            lane_list.append(np.array(single_lane))
    
        img_vis = img.copy()
        #Polylines(image,array of points, isClosed,Color of line, Thickness of the polyline edges )
        # isClosed - Flag indicating whether the drawn polylines are closed or not. 
        # If they are closed, the function draws a line from the last vertex of each curve to its first vertex.
        cv2.polylines(img_vis,np.array(lane_list),False,(0,255,255),3)


        ######## Preparing and Saving Binary File ##########
        path_binary_updated = path_binary + str(file_name_counter) + ext  

        img_bin = np.zeros([img_vis.shape[0],img_vis.shape[1],3],dtype=np.uint8)
        img_bin.fill(0)
        img_bin = cv2.cvtColor(img_bin, cv2.COLOR_BGR2GRAY)
        cv2.polylines(img_bin,np.array(lane_list),False,(255,255,255),5)
        plt.imsave(path_binary_updated,img_bin,cmap = "gray")
        print(img_bin.shape)

        ######## Preparing and Saving Instance File ##########
        path_instance_updated = path_instance + str(file_name_counter) + ext  
        
        img_ins = np.zeros([img_vis.shape[0],img_vis.shape[1],3],dtype=np.uint8)
        img_ins.fill(0)
        img_ins = cv2.cvtColor(img_ins, cv2.COLOR_BGR2GRAY)

        # Gray Color Variemnt used for instance file
        #170,170,170 - 120,120,120 - 70,70,70 - 20,20,20
        color = 20
        for l in lane_list:
            cv2.polylines(img_ins,[l],False,(color,color,color),5)
            color +=50
        plt.imsave(path_instance_updated,img_ins,cmap = "gray")
        print(img_ins.shape)

        #In parallel, writting the train.txt file that contails the link of images created
        str_path = str_img + str(file_name_counter) + ".jpg" +  " " + str_bin + str(file_name_counter) + ".jpg" + " " + str_ins + str(file_name_counter) + ".jpg" 
        outF.write(str_path)
        outF.write("\n")
        file_name_counter += 1

