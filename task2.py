import cv2 as cv
import os
import numpy as np
import pdb
import ntpath
import glob

import numpy as np
from sklearn.svm import LinearSVC, SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import glob
import cv2 as cv
import pdb
import pickle
import ntpath
from copy import deepcopy
import timeit
from skimage.feature import hog
from tqdm import tqdm
from collections import defaultdict
from skimage.transform import pyramid_gaussian

import pdb
import os
from datetime import datetime

class Parameters:
    def __init__(self):
        
        # kaggle
        #self.base_dir = '/kaggle/input'
        self.base_dir = './'
        
        self.dir_train_examples = os.path.join(self.base_dir, 'antrenare')
        self.dir_test_examples = os.path.join(self.base_dir,'testare')
        
        self.use_small_batch = False 
        self.path_annotations = os.path.join(self.base_dir, 'validare/task1_gt_validare.txt')
        
        self.path_annotations_characters = {'dad': None, 'mom': None, 'dexter': None, 'deedee': None}
        for character in ['dad', 'mom', 'dexter', 'deedee']:
            self.path_annotations_characters[character] = os.path.join(self.base_dir, f'validare/task2_{character}_gt_validare.txt')
        
        # kaggle
        #self.dir_save_files = os.path.join('/kaggle/working/', 'salveazaFisiere')
        self.dir_save_files = os.path.join('./', 'salveazaFisiere')
        
        
        if not os.path.exists(self.dir_save_files):
            os.makedirs(self.dir_save_files)
            print('directory created: {} '.format(self.dir_save_files))
        else:
            print('directory {} exists '.format(self.dir_save_files))

        # set the parameters
        # TODO: aici poate facem separat pentru fiecare personaj in parte
        self.dim_window = 36  # exemplele pozitive (fete de oameni cropate) au 36x36 pixeli
        self.dim_hog_cell = 6  # dimensiunea celulei
        self.overlap = 0.3
        self.number_positive_examples = 6713  # numarul exemplelor pozitive
        self.number_negative_examples = 10000  # numarul exemplelor negative
        self.has_annotations = False
        self.threshold = 0
        self.use_hard_mining = True  # (optional) antrenare cu exemple puternic negative
        self.use_flip_images = True  # adaugam imaginile cu fete oglindite

        self.characters = ['dad', 'mom', 'dexter', 'deedee', 'face']
        self.images_faces = {} # for each image, extract the identified faces - (x_min, y_min, x_max, y_max)
        self.characters_faces = {} # for each character, extract their identified faces and the image path
        self.character_aspect_ratio = {'dad': 0.85, 'mom': 0.90, 'dexter': 1.25, 'deedee': 1.45, 'face': 1, 'unknown': 1} # dif x / dif y = width / height
        # kaggle
        self.dir_hard_mining = os.path.join('/kaggle/working/', 'hard_mining')
        # self.windows = ['w_0.75', 'w_1', 'w_1.25', 'w_1.5']
        # self.window_aspect_ratio = {'w_0.75': [1.33, 1.0], 'w_1': [1.0, 1.0], 'w_1.25': [1.0, 1.25], 'w_1.5': [1.0, 1.5]}
        self.windows = ['dad', 'mom', 'dexter', 'deedee']
        self.window_aspect_ratio = {'dad': [4/3, 1.0], 'mom': [1.0, 1.0], 'dexter': [1.0, 1.25], 'deedee': [1.0, 1.5]}
        self.window_threshold = {'dad': 0, 'mom': 0, 'dexter': 0, 'deedee': 0}

        # kaggle
        # self.final_save_dir = '/kaggle/working/351_Ciuperceanu_Vlad'
        # self.task1_save_dir = '/kaggle/working/351_Ciuperceanu_Vlad/task1'
        # self.task2_save_dir = '/kaggle/working/351_Ciuperceanu_Vlad/task2'
        self.final_save_dir = './351_Ciuperceanu_Vlad'
        self.task1_save_dir = './351_Ciuperceanu_Vlad/task1'
        self.task2_save_dir = './351_Ciuperceanu_Vlad/task2'
        if not os.path.exists(self.final_save_dir):
            os.makedirs(self.final_save_dir)
        if not os.path.exists(self.task1_save_dir):
            os.makedirs(self.task1_save_dir)
        if not os.path.exists(self.task2_save_dir):
            os.makedirs(self.task2_save_dir)


def show_detections_without_ground_truth(detections, scores, file_names, params: Parameters, character):
    """
    Afiseaza si salveaza imaginile adnotate.
    detections: numpy array de dimensiune NX4, unde N este numarul de detectii pentru toate imaginile.
    detections[i, :] = [x_min, y_min, x_max, y_max]
    scores: numpy array de dimensiune N, scorurile pentru toate detectiile pentru toate imaginile.
    file_names: numpy array de dimensiune N, pentru fiecare detectie trebuie sa salvam numele imaginii.
    (doar numele, nu toata calea).
    """
    test_images_path = os.path.join(params.dir_test_examples, '*.jpg')
    test_files = glob.glob(test_images_path)
    test_files = sorted(test_files)

    for test_file in tqdm(test_files):
        image = cv.imread(test_file)
        short_file_name = ntpath.basename(test_file)
        indices_detections_current_image = np.where(file_names == short_file_name)
        current_detections = detections[indices_detections_current_image]
        current_scores = scores[indices_detections_current_image]

        for idx, detection in enumerate(current_detections):
            cv.rectangle(image, (detection[0], detection[1]), (detection[2], detection[3]), (0, 0, 255), thickness=1)
            cv.putText(image, 'score:' + str(current_scores[idx])[:4], (detection[0], detection[1]),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        detections_folder = os.path.join(params.dir_save_files, f"detections_{character}_{params.dim_window}_{params.dim_hog_cell}_{params.use_hard_mining}")
        if not os.path.exists(detections_folder):
            os.makedirs(detections_folder)
        cv.imwrite(os.path.join(detections_folder, f"detections_{character}_" + short_file_name), image)
        #print('Apasa orice tasta pentru a continua...')
        #cv.imshow('image', np.uint8(image))
        #cv.waitKey(0)

def show_detections_with_ground_truth(detections, scores, file_names, params: Parameters, character):
    """
    Afiseaza si salveaza imaginile adnotate. Deseneaza bounding box-urile prezice si cele corecte.
    detections: numpy array de dimensiune NX4, unde N este numarul de detectii pentru toate imaginile.
    detections[i, :] = [x_min, y_min, x_max, y_max]
    scores: numpy array de dimensiune N, scorurile pentru toate detectiile pentru toate imaginile.
    file_names: numpy array de dimensiune N, pentru fiecare detectie trebuie sa salvam numele imaginii.
    (doar numele, nu toata calea).
    """
    ground_truth_path = params.path_annotations_characters[character]
    if params.use_small_batch:
        ground_truth_path = ground_truth_path.replace('.txt', '20.txt')
    ground_truth_bboxes = np.loadtxt(ground_truth_path, dtype='str')
    test_images_path = os.path.join(params.dir_test_examples, '*.jpg')
    test_files = glob.glob(test_images_path)
    test_files = sorted(test_files)
    if params.use_small_batch:
        test_files = test_files[:20]

    for test_file in tqdm(test_files):
        image = cv.imread(test_file)
        short_file_name = ntpath.basename(test_file)
        indices_detections_current_image = np.where(file_names == short_file_name)
        current_detections = detections[indices_detections_current_image]
        current_scores = scores[indices_detections_current_image]

        for idx, detection in enumerate(current_detections):
            cv.rectangle(image, (detection[0], detection[1]), (detection[2], detection[3]), (0, 0, 255), thickness=1)
            cv.putText(image, 'score:' + str(current_scores[idx])[:4], (detection[0], detection[1]),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        annotations = ground_truth_bboxes[ground_truth_bboxes[:, 0] == short_file_name]

        # show ground truth bboxes
        for detection in annotations:
            cv.rectangle(image, (int(detection[1]), int(detection[2])), (int(detection[3]), int(detection[4])), (0, 255, 0), thickness=1)

        detections_folder = os.path.join(params.dir_save_files, f"detections_{character}_{params.dim_window}_{params.dim_hog_cell}_{params.use_hard_mining}")
        if not os.path.exists(detections_folder):
            os.makedirs(detections_folder)
        cv.imwrite(os.path.join(detections_folder, f"detections_{character}" + short_file_name), image)
        #print('Apasa orice tasta pentru a continua...')
        #cv.imshow('image', np.uint8(image))
        #cv.waitKey(0)

class FacialDetector:
    def __init__(self, params:Parameters):
        self.params = params
        all_possible_windows = []
        all_possible_windows.extend(self.params.windows)
        all_possible_windows.extend(self.params.characters)
        all_possible_windows.append('single')
        self.best_models = {window: None for window in all_possible_windows}

    def read_annotations(self, file_path):
        image_faces = defaultdict(list)
        character_faces = defaultdict(list)
        character_faces['face'] = []
        character = ntpath.basename(file_path).split('_')[0]

        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 6:
                    continue

                image_name, xmin, ymin, xmax, ymax, character_found = parts
                xmin, ymin, xmax, ymax = map(int, [xmin, ymin, xmax, ymax])
                image_path = os.path.join(self.params.dir_train_examples, character, image_name)
                # note that this will also contain 'unknown'
                if character_found not in character_faces:
                    character_faces[character_found] = [(xmin, ymin, xmax, ymax, image_path)]
                else:
                    character_faces[character_found].append((xmin, ymin, xmax, ymax, image_path))
                character_faces['face'].append((xmin, ymin, xmax, ymax, image_path)) # we keep all the faces in a separate list
                if image_path not in image_faces:
                    image_faces[image_path] = [(xmin, ymin, xmax, ymax, character_found)]
                else:
                    image_faces[image_path].append((xmin, ymin, xmax, ymax, character_found))

        return image_faces, character_faces

    def get_images_faces(self):
        for character in self.params.characters:
            if character == 'face':
                continue
            character_annotations_path = os.path.join(self.params.dir_train_examples, character + '_annotations.txt')
            image_faces, character_faces = self.read_annotations(character_annotations_path)

            for key, value in image_faces.items():
                if key not in self.params.images_faces:
                    self.params.images_faces[key] = value
                else:
                    self.params.images_faces[key].extend(value)
            for key, value in character_faces.items():
                if key not in self.params.characters_faces:
                    self.params.characters_faces[key] = value
                else:
                    self.params.characters_faces[key].extend(value)
    
    def get_positive_descriptors_per_window(self, window):
        positive_descriptors = []
        character_faces = self.params.characters_faces[window] # take only specific character faces
        for face in tqdm(character_faces):
            image_path = face[4]
            img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

            # extract the actual face from the image and resize it to the desired size
            face_coordinates = face[:4]
            #img = img[face_coordinates[1]:face_coordinates[3], face_coordinates[0]:face_coordinates[2]]
            
            face_width = face_coordinates[2] - face_coordinates[0]
            face_height = face_coordinates[3] - face_coordinates[1]
            
            extra_width = int(face_width * 0.10)
            extra_height = int(face_height * 0.10)
            
            new_xmin = max(face_coordinates[0] - extra_width, 0)  # Asigură-te că xmin nu devine negativ
            new_ymin = max(face_coordinates[1] - extra_height, 0)  # Asigură-te că ymin nu devine negativ
            new_xmax = min(face_coordinates[2] + extra_width, img.shape[1])
            new_ymax = min(face_coordinates[3] + extra_height, img.shape[0])
            
            img = img[new_ymin:new_ymax, new_xmin:new_xmax]
            x_size = int(self.params.dim_window * self.params.window_aspect_ratio[window][1])
            y_size = int(self.params.dim_window * self.params.window_aspect_ratio[window][0])

            bigger_img = cv.resize(img, None, fx=1.1, fy=1.1, interpolation=cv.INTER_LINEAR)

            for resized_image in pyramid_gaussian(bigger_img, downscale=1.2, preserve_range=True):
                if resized_image.shape[0] < y_size or resized_image.shape[1] < x_size:
                    break

                resized_image_for_hog = cv.resize(resized_image, (x_size, y_size))
                features = hog(resized_image_for_hog, pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
                               cells_per_block=(2, 2), feature_vector=True)
                positive_descriptors.append(features)
                if self.params.use_flip_images:
                    features = hog(np.fliplr(resized_image_for_hog), pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell), cells_per_block=(2, 2), feature_vector=True)
                    positive_descriptors.append(features)
            
            img = cv.resize(img, (x_size, y_size))
            features = hog(img, pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell), cells_per_block=(2, 2), feature_vector=True)
            positive_descriptors.append(features)
            if self.params.use_flip_images:
                features = hog(np.fliplr(img), pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell), cells_per_block=(2, 2), feature_vector=True)
                positive_descriptors.append(features)

        positive_descriptors = np.array(positive_descriptors)
        return positive_descriptors

    def intersect(self, box1, box2):
        if not (box1[2] < box2[0] or box2[2] < box1[0] or box1[3] < box2[1] or box2[3] < box1[1]):
            return True
        return False        

    def get_negative_descriptors_for_characters(self, character):
        negative_descriptors = []
        
        # here extract half of the total negative examples from the other characters
        all_other_characters_faces = []
        for other_character in self.params.characters:
            if other_character == 'face' or other_character == character:
                continue
            all_other_characters_faces.extend(self.params.characters_faces[other_character])
        all_other_characters_faces.extend(self.params.characters_faces['unknown']) # we also add the unknown faces

        available_faces = 0
        for face in all_other_characters_faces:
            image_path = face[4]
            img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

            # extract the actual face from the image
            face_coordinates = face[:4]
            face_width = face_coordinates[2] - face_coordinates[0]
            face_height = face_coordinates[3] - face_coordinates[1]
            
            extra_width = int(face_width * 0.5) # TODO: marim mai mult bounding box-ul ca sa ia mai mult
            extra_height = int(face_height * 0.5)
            
            new_xmin = max(face_coordinates[0] - extra_width, 0)  # Asigură-te că xmin nu devine negativ
            new_ymin = max(face_coordinates[1] - extra_height, 0)  # Asigură-te că ymin nu devine negativ
            new_xmax = min(face_coordinates[2] + extra_width, img.shape[1])
            new_ymax = min(face_coordinates[3] + extra_height, img.shape[0])
            
            img = img[new_ymin:new_ymax, new_xmin:new_xmax]
            x_size = int(self.params.window_aspect_ratio[character][1] * self.params.dim_window)
            y_size = int(self.params.window_aspect_ratio[character][0] * self.params.dim_window)

            num_rows = img.shape[0]
            num_cols = img.shape[1]

            if num_rows <= y_size or num_cols <= x_size:
                continue
            available_faces += 1

        num_negative_per_image = (self.params.number_negative_examples // 2) // available_faces
        print('Calculam descriptorii pt %d imagini negative pentru %s' % (len(all_other_characters_faces), character))
        for face in tqdm(all_other_characters_faces):
            image_path = face[4]
            img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

            # extract the actual face from the image
            face_coordinates = face[:4]
            face_width = face_coordinates[2] - face_coordinates[0]
            face_height = face_coordinates[3] - face_coordinates[1]
            
            extra_width = int(face_width * 0.5)
            extra_height = int(face_height * 0.5)
            
            new_xmin = max(face_coordinates[0] - extra_width, 0)  # Asigură-te că xmin nu devine negativ
            new_ymin = max(face_coordinates[1] - extra_height, 0)  # Asigură-te că ymin nu devine negativ
            new_xmax = min(face_coordinates[2] + extra_width, img.shape[1])
            new_ymax = min(face_coordinates[3] + extra_height, img.shape[0])
            
            img = img[new_ymin:new_ymax, new_xmin:new_xmax]
            x_size = int(self.params.window_aspect_ratio[character][1] * self.params.dim_window)
            y_size = int(self.params.window_aspect_ratio[character][0] * self.params.dim_window)

            num_rows = img.shape[0]
            num_cols = img.shape[1]

            if num_rows <= y_size or num_cols <= x_size:
                continue

            x = np.random.randint(low=0, high=num_cols - x_size, size=num_negative_per_image)
            y = np.random.randint(low=0, high=num_rows - y_size, size=num_negative_per_image)

            for idx in range(len(y)):
                patch = img[y[idx]: y[idx] + y_size, x[idx]: x[idx] + x_size]
                descr = hog(patch, pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell), cells_per_block=(2, 2), feature_vector=False)
                negative_descriptors.append(descr.flatten())
                
                
        # now we take the other half from random patches
        files = []
        for ch in self.params.characters:
            if ch == 'face':
                continue
            character_images_path = os.path.join(self.params.dir_train_examples, ch, '*.jpg')
            character_files = glob.glob(character_images_path)
            files.extend(character_files)

        num_images = len(files)
        num_negative_per_image = (self.params.number_negative_examples // 2) // num_images
        print('Calculam descriptorii pt %d imagini negative' % num_images)
        for i in tqdm(range(num_images)):
            img = cv.imread(files[i], cv.IMREAD_GRAYSCALE)
            num_rows = img.shape[0]
            num_cols = img.shape[1]

            # we need to check if the negative patch overlaps with a positive patch (a face)
            x = [] # TODO: maybe check if the patch is already in the list; maybe we need to take iou
            y = [] # into account if it's not possible by some reasons to find a patch that doesn't intersect
            # try to let him include negative patches of other people's faces
            # TODO: or maybe leave it by not allowing the patch to intersect with any face

            x_size = int(self.params.window_aspect_ratio[character][1] * self.params.dim_window)
            y_size = int(self.params.window_aspect_ratio[character][0] * self.params.dim_window)

            for j in range(num_negative_per_image):
                # we need to choose a random patch which does not overlap with a face
                found = False
                
                if not found:
                    for _ in range(30):
                        x_rand = np.random.randint(low=0, high=num_cols - x_size)
                        y_rand = np.random.randint(low=0, high=num_rows - y_size)
                        patch_coordinates = np.array([x_rand, y_rand, x_rand + x_size, y_rand + y_size])
                        intersects = False
                        for face_found in self.params.images_faces[files[i]]:
                            face_coordinates = np.array(face_found[:4])
                            if self.intersection_over_union(patch_coordinates, face_coordinates) >= 0.15: # TODO: parametru invatat
                                intersects = True
                                break
                        
                        if not intersects:
                            x.append(x_rand)
                            y.append(y_rand)
                            found = True
                            break
                
                if not found:
                    x_rand = np.random.randint(low=0, high=num_cols - x_size)
                    y_rand = np.random.randint(low=0, high=num_rows - y_size)
                    x.append(x_rand)
                    y.append(y_rand)
                

            x = np.array(x)
            y = np.array(y)

            for idx in range(len(y)):
                patch = img[y[idx]: y[idx] + y_size, x[idx]: x[idx] + x_size]
                descr = hog(patch, pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell), cells_per_block=(2, 2), feature_vector=False)
                negative_descriptors.append(descr.flatten())

        negative_descriptors = np.array(negative_descriptors)
        return negative_descriptors

    def train_classifier(self, training_examples, train_labels, character):
        hard_mining_used = 'with_hard_mining' if self.params.use_hard_mining else 'without_hard_mining'
        svm_file_name = os.path.join(self.params.dir_save_files, f'best_model_rbf_pyramid_{character}_{hard_mining_used}_{self.params.dim_window}_{self.params.dim_hog_cell}_{self.params.number_negative_examples}_{self.params.number_positive_examples}.pkl')
        
        if os.path.exists(svm_file_name):
            print('Incarcam clasificatorul pentru: %s' % character)
            #print('De la locatia: %s' % svm_file_name)
            with open(svm_file_name, 'rb') as f:
                self.best_models[character] = pickle.load(f)
            return

        print('Antrenam clasificatorul pentru: %s' % character)
        best_accuracy = 0
        best_c = 1
        best_model = None
        Cs = [10 ** -5, 10 ** -4,  10 ** -3,  10 ** -2, 10 ** -1, 10 ** 0]
        # for c in Cs:
        #     print('Antrenam un clasificator pentru c=%f' % c)
        model = SVC(C=1, kernel='rbf', verbose=True)
        model.fit(training_examples, train_labels)
            # acc = model.score(training_examples, train_labels)
            # print(acc)
            # if acc > best_accuracy:
            #     best_accuracy = acc
            #     best_c = c
            #     best_model = deepcopy(model)

        print('Performanta clasificatorului optim pt c = %f' % best_c)
        # salveaza clasificatorul
        with open(svm_file_name, 'wb') as f:
            pickle.dump(model, f)

        # vizualizeaza cat de bine sunt separate exemplele pozitive de cele negative dupa antrenare
        # ideal ar fi ca exemplele pozitive sa primeasca scoruri > 0, iar exemplele negative sa primeasca scoruri < 0
        #scores = best_model.decision_function(training_examples)
        self.best_models[character] = model
        #positive_scores = scores[train_labels > 0]
        #negative_scores = scores[train_labels <= 0]


        # plt.plot(np.sort(positive_scores))
        # plt.plot(np.zeros(len(positive_scores)))
        # plt.plot(np.sort(negative_scores))
        # plt.xlabel('Nr example antrenare')
        # plt.ylabel('Scor clasificator')
        # plt.title('Distributia scorurilor clasificatorului pe exemplele de antrenare pentru %s' % character)
        # plt.legend(['Scoruri exemple pozitive', '0', 'Scoruri exemple negative'])
        # plt.show()

    def intersection_over_union(self, bbox_a, bbox_b):
        x_a = max(bbox_a[0], bbox_b[0])
        y_a = max(bbox_a[1], bbox_b[1])
        x_b = min(bbox_a[2], bbox_b[2])
        y_b = min(bbox_a[3], bbox_b[3])

        inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)

        box_a_area = (bbox_a[2] - bbox_a[0] + 1) * (bbox_a[3] - bbox_a[1] + 1)
        box_b_area = (bbox_b[2] - bbox_b[0] + 1) * (bbox_b[3] - bbox_b[1] + 1)

        if float(box_a_area + box_b_area - inter_area) == 0:
            print('Found division by zero!')
            print('Box a and box b', bbox_a, bbox_b)
            print('coord xa ya xb yb', x_a, y_a, x_b, y_b)
            print('inter area, box_a_area, box_b_area', inter_area, box_a_area, box_b_area)
            return 0.0

        iou = inter_area / float(box_a_area + box_b_area - inter_area)

        return iou

    def non_maximal_suppression(self, image_detections, image_scores, image_size, image_predictions=None, image_file_names=None):
        """
        Detectiile cu scor mare suprima detectiile ce se suprapun cu acestea dar au scor mai mic.
        Detectiile se pot suprapune partial, dar centrul unei detectii nu poate
        fi in interiorul celeilalte detectii.
        :param image_detections:  numpy array de dimensiune NX4, unde N este numarul de detectii.
        :param image_scores: numpy array de dimensiune N
        :param image_size: tuplu, dimensiunea imaginii
        :return: image_detections si image_scores care sunt maximale.
        """

        # xmin, ymin, xmax, ymax
        x_out_of_bounds = np.where(image_detections[:, 2] > image_size[1])[0]
        y_out_of_bounds = np.where(image_detections[:, 3] > image_size[0])[0]
        if len(x_out_of_bounds) > 0:
            print(image_detections[x_out_of_bounds, 2], image_detections[y_out_of_bounds, 3])
            print(image_size[1], image_size[0])
        image_detections[x_out_of_bounds, 2] = image_size[1]
        image_detections[y_out_of_bounds, 3] = image_size[0]

        sorted_indices = np.flipud(np.argsort(image_scores))
        sorted_image_detections = image_detections[sorted_indices]
        sorted_scores = image_scores[sorted_indices]
        if image_predictions is not None:
            sorted_image_predictions = image_predictions[sorted_indices]
        if image_file_names is not None:
            sorted_image_file_names = image_file_names[sorted_indices]

        is_maximal = np.ones(len(image_detections)).astype(bool)
        iou_threshold = 0.3
        for i in range(len(sorted_image_detections) - 1):
            if is_maximal[i] == True:  # don't change to 'is True' because is a numpy True and is not a python True :)
                for j in range(i + 1, len(sorted_image_detections)):
                    if is_maximal[j] == True:  # don't change to 'is True' because is a numpy True and is not a python True :)
                        if self.intersection_over_union(sorted_image_detections[i],sorted_image_detections[j]) > iou_threshold:
                            is_maximal[j] = False
                        else:  # verificam daca centrul detectiei este in mijlocul detectiei cu scor mai mare
                            c_x = (sorted_image_detections[j][0] + sorted_image_detections[j][2]) / 2
                            c_y = (sorted_image_detections[j][1] + sorted_image_detections[j][3]) / 2
                            if sorted_image_detections[i][0] <= c_x <= sorted_image_detections[i][2] and \
                                    sorted_image_detections[i][1] <= c_y <= sorted_image_detections[i][3]:
                                is_maximal[j] = False
        if image_predictions is not None and image_file_names is not None:
            return sorted_image_detections[is_maximal], sorted_scores[is_maximal], sorted_image_file_names[is_maximal], sorted_image_predictions[is_maximal] 
        elif image_predictions is not None:
            return sorted_image_detections[is_maximal], sorted_scores[is_maximal], sorted_image_predictions[is_maximal]
        elif image_file_names is not None:
            return sorted_image_detections[is_maximal], sorted_scores[is_maximal], sorted_image_file_names[is_maximal]
        else:
            return sorted_image_detections[is_maximal], sorted_scores[is_maximal]
    
    def predict_face(self, img, initial_resize_factor, window):
        image_scores = np.array([])
        image_detections = None
        upscale_factor = 1.0
        
        x_size = int(self.params.window_aspect_ratio[window][1] * self.params.dim_window)
        y_size = int(self.params.window_aspect_ratio[window][0] * self.params.dim_window)

        for resized_image in pyramid_gaussian(img, downscale=1.05, preserve_range=True):
            if resized_image.shape[0] < y_size or resized_image.shape[1] < x_size:
                break

            hog_descriptors = hog(resized_image, pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell), cells_per_block=(2, 2), feature_vector=False)
            num_rows = resized_image.shape[0] // self.params.dim_hog_cell - 1
            num_cols = resized_image.shape[1] // self.params.dim_hog_cell - 1

            num_cell_in_template_y = y_size // self.params.dim_hog_cell - 1
            num_cell_in_template_x = x_size // self.params.dim_hog_cell - 1

            resized_image_scores = []
            resized_image_detections = []

            for y in range(0, num_rows - num_cell_in_template_y):
                for x in range(0, num_cols - num_cell_in_template_x):
                    hog_descriptor = hog_descriptors[y:y + num_cell_in_template_y, x:x + num_cell_in_template_x].flatten()
                    score = self.best_models[window].decision_function(hog_descriptor.reshape(1, -1))[0]
                    if score > self.params.window_threshold[window]:
                        x_min = int(x * self.params.dim_hog_cell * upscale_factor / initial_resize_factor)
                        y_min = int(y * self.params.dim_hog_cell * upscale_factor / initial_resize_factor)
                        x_max = int((x * self.params.dim_hog_cell + x_size) * upscale_factor / initial_resize_factor)
                        y_max = int((y * self.params.dim_hog_cell + y_size) * upscale_factor / initial_resize_factor)
                        resized_image_scores.append(score) # we do this so we actually keep the normal boxes
                        resized_image_detections.append([x_min, y_min, x_max, y_max])
                        #descriptors_to_return.append(hog_descriptor)

            if len(resized_image_scores) > 0:
                resized_image_detections, resized_image_scores = self.non_maximal_suppression(np.array(resized_image_detections), np.array(resized_image_scores), [5000, 5000])

            if len(resized_image_scores) > 0:
                if image_detections is None:
                    image_detections = resized_image_detections
                else:
                    image_detections = np.concatenate((image_detections, resized_image_detections))
                image_scores = np.append(image_scores, resized_image_scores)

            upscale_factor *= 1.05

        # we use non-maximal suppression again, since we used multi-scale detection
        if len(image_scores) > 0:
            image_detections, image_scores = self.non_maximal_suppression(np.array(image_detections), np.array(image_scores), [5000, 5000])
            # TODO: we put bigger image size in nms since it would make detections seem out of bounds, maybe change that (we still go from a bigger image now)
            
        return image_detections, image_scores
    
    def run(self, small_batch=False):

        test_images_path = os.path.join(self.params.dir_test_examples, '*.jpg')
        test_files = glob.glob(test_images_path)
        test_files = sorted(test_files)
        if small_batch:
            test_files = test_files[:20]
        detections = None  # array cu toate detectiile pe care le obtinem
        scores = np.array([])  # array cu toate scorurile pe care le obtinem
        file_names = np.array([])  # array cu fisierele, in aceasta lista fisierele vor aparea de mai multe ori, pentru fiecare detectie din imagine, numele imaginii va aparea in aceasta lista
        predictions = np.array([])
        num_test_images = len(test_files)

        detections_per_window = {}
        scores_per_window = {}
        file_names_per_window = {}
        for window in self.params.windows:
            detections_per_window[window] = None
            scores_per_window[window] = np.array([])
            file_names_per_window[window] = np.array([])

        for i in tqdm(range(num_test_images)):
            img = cv.imread(test_files[i], cv.IMREAD_GRAYSCALE)
            
            windows_detections = None
            windows_scores = np.array([])
            windows_file_names = np.array([])
            windows_predictions = np.array([])
            
            try:
                for window in self.params.windows:
                    
                    initial_resize_factor = 1.2
                    if window == 'deedee':
                        initial_resize_factor = 1
                    bigger_img = cv.resize(img, None, fx=initial_resize_factor, fy=initial_resize_factor, interpolation=cv.INTER_LINEAR)
                    
                    image_detections, image_scores = self.predict_face(bigger_img, initial_resize_factor, window)
                
                    if len(image_scores) > 0:
                        if windows_detections is None:
                            windows_detections = image_detections
                        else:
                            windows_detections = np.concatenate((windows_detections, image_detections))
                        windows_scores = np.append(windows_scores, image_scores)
                        image_names = [ntpath.basename(test_files[i]) for _ in range(len(image_scores))]
                        windows_file_names = np.append(windows_file_names, image_names)
                        windows_predictions = np.append(windows_predictions, [window for _ in range(len(image_scores))])

                        if detections_per_window[window] is None:
                            detections_per_window[window] = image_detections
                        else:
                            detections_per_window[window] = np.concatenate((detections_per_window[window], image_detections))
                        scores_per_window[window] = np.append(scores_per_window[window], image_scores)
                        file_names_per_window[window] = np.append(file_names_per_window[window], image_names)
            except Exception as e:
                print('Exceptie la imaginea %s' % test_files[i])
                print(e)
                continue    
            # no more nms since they are diff characters
            # if len(windows_scores) > 0:
            #     #print('predictions before nms', windows_predictions)
            #     windows_detections, windows_scores, windows_file_names, windows_predictions = self.non_maximal_suppression(np.array(windows_detections), np.array(windows_scores), img.shape, image_predictions=np.array(windows_predictions), image_file_names=np.array(windows_file_names))
            
            if len(windows_scores) > 0:
                if detections is None:
                    detections = windows_detections
                else:
                    detections = np.concatenate((detections, windows_detections))
                scores = np.append(scores, windows_scores)
                file_names = np.append(file_names, windows_file_names)
                predictions = np.append(predictions, windows_predictions)
                #print(predictions)

        return detections, scores, file_names, predictions, detections_per_window, scores_per_window, file_names_per_window

    def compute_average_precision(self, rec, prec):
        # functie adaptata din 2010 Pascal VOC development kit
        m_rec = np.concatenate(([0], rec, [1]))
        m_pre = np.concatenate(([0], prec, [0]))
        for i in range(len(m_pre) - 1, -1, 1):
            m_pre[i] = max(m_pre[i], m_pre[i + 1])
        m_rec = np.array(m_rec)
        i = np.where(m_rec[1:] != m_rec[:-1])[0] + 1
        average_precision = np.sum((m_rec[i] - m_rec[i - 1]) * m_pre[i]) # aria de sub curba, calc ca suma ariilor dreptunghiurilor
        return average_precision

    def eval_detections_character(self, detections, scores, file_names,ground_truth_path,character):
        if self.params.use_small_batch:
            ground_truth_path = ground_truth_path.replace('.txt', '20.txt')
        ground_truth_file = np.loadtxt(ground_truth_path, dtype='str')
        ground_truth_file_names = np.array(ground_truth_file[:, 0])
        ground_truth_detections = np.array(ground_truth_file[:, 1:], np.int32)
    
        num_gt_detections = len(ground_truth_detections)  # numar total de adevarat pozitive
        gt_exists_detection = np.zeros(num_gt_detections)
        # sorteazam detectiile dupa scorul lor
        sorted_indices = np.argsort(scores)[::-1]
        file_names = file_names[sorted_indices]
        scores = scores[sorted_indices]
        detections = detections[sorted_indices]
    
        num_detections = len(detections)
        true_positive = np.zeros(num_detections)
        false_positive = np.zeros(num_detections)
        duplicated_detections = np.zeros(num_detections)
    
        for detection_idx in range(num_detections):
            indices_detections_on_image = np.where(ground_truth_file_names == file_names[detection_idx])[0]
    
            gt_detections_on_image = ground_truth_detections[indices_detections_on_image]
            bbox = detections[detection_idx]
            max_overlap = -1
            index_max_overlap_bbox = -1
            for gt_idx, gt_bbox in enumerate(gt_detections_on_image):
                overlap = self.intersection_over_union(bbox, gt_bbox)
                if overlap > max_overlap:
                    max_overlap = overlap
                    index_max_overlap_bbox = indices_detections_on_image[gt_idx]
    
            # clasifica o detectie ca fiind adevarat pozitiva / fals pozitiva
            if max_overlap >= 0.3:
                if gt_exists_detection[index_max_overlap_bbox] == 0:
                    true_positive[detection_idx] = 1
                    gt_exists_detection[index_max_overlap_bbox] = 1
                else:
                    false_positive[detection_idx] = 1
                    duplicated_detections[detection_idx] = 1
            else:
                false_positive[detection_idx] = 1
    
        cum_false_positive = np.cumsum(false_positive)
        cum_true_positive = np.cumsum(true_positive)
    
        rec = cum_true_positive / num_gt_detections
        prec = cum_true_positive / (cum_true_positive + cum_false_positive)
        average_precision = self.compute_average_precision(rec, prec)
        plt.plot(rec, prec, '-')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'{character} faces: average precision = {average_precision}')
        plt.savefig('precizie_medie_' + character + '.png')
        plt.show()
             
    def save_task1(self, detections, scores, file_names):
        np.save(os.path.join(self.params.task1_save_dir, f'detections_all_faces.npy'), detections, allow_pickle=True, fix_imports=True)
        np.save(os.path.join(self.params.task1_save_dir, f'scores_all_faces.npy'), scores, allow_pickle=True, fix_imports=True)
        np.save(os.path.join(self.params.task1_save_dir, f'file_names_all_faces.npy'), file_names, allow_pickle=True, fix_imports=True)

    def save_task2(self, character_detections, character_scores, character_file_names, character):
        np.save(os.path.join(self.params.task2_save_dir, f'detections_{character}.npy'), character_detections, allow_pickle=True, fix_imports=True)
        np.save(os.path.join(self.params.task2_save_dir, f'scores_{character}.npy'), character_scores, allow_pickle=True, fix_imports=True)
        np.save(os.path.join(self.params.task2_save_dir, f'file_names_{character}.npy'), character_file_names, allow_pickle=True, fix_imports=True)

params: Parameters = Parameters()
params.dim_window = 90  # exemplele pozitive (fete de oameni cropate) au 36x36 pixeli
params.dim_hog_cell = 18  # dimensiunea celulei
params.overlap = 0.3
params.number_positive_examples = 4000  # numarul exemplelor pozitive
params.number_negative_examples = 48000  # numarul exemplelor negative

params.threshold = 0 # toate ferestrele cu scorul > threshold si maxime locale devin detectii
params.has_annotations = False

params.use_hard_mining = False  # (optional)antrenare cu exemple puternic negative
params.use_flip_images = True  # adauga imaginile cu fete oglindite

if params.use_flip_images:
    params.number_positive_examples *= 2

facial_detector: FacialDetector = FacialDetector(params)
facial_detector.get_images_faces()

# Pasii 1+2+3. Incarcam exemplele pozitive (cropate) si exemple negative generate
# verificam daca sunt deja existente
# TODO: check if the correct ones are loaded, maybe we need to delete the folder and run again (if modifications are made)
# this includes all positive and negative descriptors (including the ones for the characters)

for window in facial_detector.params.windows:
    positive_features_window_path = os.path.join(params.dir_save_files, 'descriptoriExemplePozitive_orientations_9_pyramid' + str(window) + '_' + str(params.dim_window) + '_' + str(params.dim_hog_cell) + '_' + str(params.number_positive_examples) + '.npy')
    if os.path.exists(positive_features_window_path):
        positive_features = np.load(positive_features_window_path, allow_pickle=True)
        print('Am incarcat descriptorii pentru exemplele pozitive pentru fereastra %s' % window)
        #print('De la locatia: %s' % positive_features_window_path)
    else:
        print('Construim descriptorii pentru exemplele pozitive pentru fereastra %s:' % window)
        positive_features = facial_detector.get_positive_descriptors_per_window(window)
        np.save(positive_features_window_path, positive_features, allow_pickle=True)
        print('Am salvat descriptorii pentru exemplele pozitive in fisierul %s' % positive_features_window_path)

    negative_features_window_path = os.path.join(params.dir_save_files, 'descriptoriExempleNegative_orientations_9_' + str(window) + '_' + str(params.dim_window) + '_' + str(params.dim_hog_cell) + '_' + str(params.number_negative_examples) + '.npy')
    if os.path.exists(negative_features_window_path):
        negative_features = np.load(negative_features_window_path, allow_pickle=True)
        print('Am incarcat descriptorii pentru exemplele negative pentru fereastra %s' % window)
        #print('De la locatia: %s' % negative_features_window_path)
    else:
        print('Construim descriptorii pentru exemplele negative pentru fereastra %s:' % window)
        negative_features = facial_detector.get_negative_descriptors_for_characters(window)
        np.save(negative_features_window_path, negative_features, allow_pickle=True)
        print('Am salvat descriptorii pentru exemplele negative in fisierul %s' % negative_features_window_path)

    training_examples = np.concatenate((np.squeeze(positive_features), np.squeeze(negative_features)), axis=0)
    train_labels = np.concatenate((np.ones(positive_features.shape[0]), np.zeros(negative_features.shape[0])))

    facial_detector.train_classifier(training_examples, train_labels, window)


# task 2
facial_detector.params.use_hard_mining = False
facial_detector.params.use_small_batch = False
params.use_small_batch = False
facial_detector.params.threshold = 0
facial_detector.params.window_threshold = {'dad': 0, 'mom': 0, 'dexter': 0, 'deedee': 0}

detections = {}
scores = {}
file_names = {}
predictions = {}
detections_per_window = {}
scores_per_window = {}
file_names_per_window = {}
detections['all'], scores['all'], file_names['all'], predictions['all'], detections_per_window, scores_per_window, file_names_per_window = facial_detector.run(small_batch=facial_detector.params.use_small_batch)

for window in facial_detector.params.windows:
    facial_detector.save_task2(detections_per_window[window], scores_per_window[window], file_names_per_window[window], window)

# task 2 - evaluate detections
if facial_detector.params.has_annotations:

    for window in facial_detector.params.windows:
        print(f'Window to be checked - {window}:')
        
        facial_detector.eval_detections_character(detections_per_window[window], scores_per_window[window], file_names_per_window[window], facial_detector.params.path_annotations_characters[window], window)
        show_detections_with_ground_truth(detections_per_window[window], scores_per_window[window], file_names_per_window[window], facial_detector.params, window)

else:
    print('No ground truth available')
    for window in facial_detector.params.windows:
        show_detections_without_ground_truth(detections_per_window[window], scores_per_window[window], file_names_per_window[window], facial_detector.params, window)