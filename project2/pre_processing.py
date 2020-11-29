import os
import numpy as np
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
from keras.applications.vgg16 import VGG16
from keras import Model
import pandas as pd

# the data in this file will be run one time and will be saved for later use

file_path = 'H:\\tamu\\courses\\MSEN660\\Homeworks\\Computer_project2\\DataSet\\CMU-UHCS_Dataset\\'


def preprocess_image(img_path, crop_y):
    img = image.load_img(img_path)
    img_arr = image.img_to_array(img)
    img_arr = img_arr[0:crop_y, :, :]
    img_arr = np.expand_dims(img_arr,
                             axis=0)  # expand image dimenssion since Keras accept images in size of (sample, size1, size2, channel)
    img_arr = preprocess_input(img_arr)  # keras pre+processing: adequate your image to the format the model requires
    return img_arr


# Notice that no image size reduction is necessary, as Keras can accept any input image size when in featurization mode.

# Transfer learning: using VGG16 as a bas_model (not including fully_connected layer) /
base_model = VGG16(weights='imagenet', include_top=False)  #
base_model.summary()
# extracting the desired feature map for the first layer of our model
model_64_b1 = Model(inputs=base_model.input, outputs=base_model.get_layer('block1_pool').input)  # block1_pool
model_128_b2 = Model(inputs=base_model.input, outputs=base_model.get_layer('block2_pool').input)  # block2_pool
model_256_b3 = Model(inputs=base_model.input, outputs=base_model.get_layer('block3_pool').input)  # block3_pool
model_512_b4 = Model(inputs=base_model.input, outputs=base_model.get_layer('block4_pool').input)  # block4_pool
model_512_b5 = Model(inputs=base_model.input, outputs=base_model.get_layer('block5_pool').input)  # block5_pool


def get_VGG16_feature(img_arr, model):
    feature_img = model.predict(img_arr)
    feature_mean = np.mean(feature_img,
                           axis=(0, 1, 2))  # note that each img_arr size is (1,image_width, image_hight, feature_map)
    return feature_mean


def model_features(file_path, model1, model2, model3, model4, model5):
    df1 = pd.read_csv(file_path + "micrograph.csv")
    df2 = pd.DataFrame(
        columns=['feature_64_b1', 'feature_128_b2', 'feature_256_b3', 'feature_512_b4', 'feature_512_b5'])

    for i in range(len(df1)):
        img_path = file_path + 'images\\' + df1.path[
            i]  # to ensure that image orders are read same as df elements (os.listdir does not garantee that)
        img_arr = preprocess_image(img_path=img_path, crop_y=484)
        feature_mean_64_b1 = get_VGG16_feature(img_arr, model1)
        feature_mean_128_b2 = get_VGG16_feature(img_arr, model2)
        feature_mean_256_b3 = get_VGG16_feature(img_arr, model3)
        feature_mean_512_b4 = get_VGG16_feature(img_arr, model4)
        feature_mean_512_b5 = get_VGG16_feature(img_arr, model5)

        df2 = df2.append({'feature_64_b1': feature_mean_64_b1,
                          'feature_128_b2': feature_mean_128_b2,
                          'feature_256_b3': feature_mean_256_b3,
                          'feature_512_b4': feature_mean_512_b4,
                          'feature_512_b5': feature_mean_512_b5}, ignore_index=True)

    df = df1.join(df2)
    df.to_csv(file_path + 'micrograph_features.csv', index=False)  # save the file as csv


# save the computed feature as a .csv file (only onetime since computation is expensive)
model_features(file_path, model_64_b1, model_128_b2, model_256_b3, model_512_b4, model_512_b5)

df = pd.read_csv(file_path + "micrograph_features.csv")

### encoding labels
all_labels = df.primary_microconstituent.unique()

conditions = [(df['primary_microconstituent'] == 'spheroidite'), (df['primary_microconstituent'] == 'network'),
              (df['primary_microconstituent'] == 'pearlite'),
              (df['primary_microconstituent'] == 'spheroidite+widmanstatten'),
              (df['primary_microconstituent'] == 'martensite'),
              (df['primary_microconstituent'] == 'pearlite+spheroidite'),
              (df['primary_microconstituent'] == 'pearlite+widmanstatten')]

values = [0, 1, 2, 3, 4, 5, 6]

df['labels'] = np.select(conditions, values)
df.to_csv(file_path + 'micrograph_features_labels.csv', index=False)

df = pd.read_csv(file_path + "micrograph_features_labels.csv")
labels = ['spheroidite', 'network', 'pearlite', 'spheroidite+widmanstatten']  # excluding the 3 noise labels

split1 = 100
split2 = 60

df_sph_train = df[df['primary_microconstituent'] == labels[0]][:split1]
df_net_train = df[df['primary_microconstituent'] == labels[1]][:split1]
df_pear_train = df[df['primary_microconstituent'] == labels[2]][:split1]
df_sph_widm_train = df[df['primary_microconstituent'] == labels[3]][:split2]

df_train = pd.concat([df_sph_train, df_net_train, df_pear_train, df_sph_widm_train], axis=0)
# get the rest of the data for test
df_all_test = df.loc[df.index.difference(df_train.index)]  # including all labels (noise labels included)
df_test = df_all_test[df_all_test['primary_microconstituent'].isin(labels)]  # getting test excluding the noise labels

# save the tratin and test data
df_train.to_csv(file_path + 'train_data.csv', index=False)  # 4 labels
df_test.to_csv(file_path + 'test_data.csv', index=False)  # 4 labels
df_all_test.to_csv(file_path + 'test_all.csv', index=False)  # total 7 labels

print(all_labels)
print('len(df_train)= ', len(df_train))
print('len(df_test)= ', len(df_test))
print('len(df_all_test)= ', len(df_all_test))
print('all_labels= ', all_labels)
print('test_labels=', df_test.primary_microconstituent.unique())  # to ensure the test data not include nose labels

'''
# split the train and test based on the problem request and save the .csv files
df = pd.read_csv(file_path + "micrograph_features.csv")

### encoding labels
all_labels = df.primary_microconstituent.unique()

conditions = [(df['primary_microconstituent']== 'spheroidite'), (df['primary_microconstituent']== 'network'),
              (df['primary_microconstituent']== 'pearlite'), (df['primary_microconstituent']== 'spheroidite+widmanstatten'),
              (df['primary_microconstituent']== 'martensite'), (df['primary_microconstituent']== 'pearlite+spheroidite'),
              (df['primary_microconstituent']== 'pearlite+widmanstatten')]

values = [0, 1, 2, 3, 4, 5, 6]

#df.to_csv(file_path + 'micrograph_features_new.csv', index=False)

df = pd.read_csv(file_path + "micrograph_features_new.csv")
labels = ['spheroidite', 'network', 'pearlite', 'spheroidite+widmanstatten']    # excluding the 3 noise labels

split1 = 100
split2 = 60

df_sph_train = df[df['primary_microconstituent']==labels[0]][:split1]
df_net_train = df[df['primary_microconstituent']==labels[1]][:split1]
df_pear_train = df[df['primary_microconstituent']==labels[2]][:split1]
df_sph_widm_train = df[df['primary_microconstituent']==labels[3]][:split2]

df_train = pd.concat([df_sph_train, df_net_train, df_pear_train, df_sph_widm_train], axis=0)
# get the rest of the data for test
df_all_test = df.loc[df.index.difference(df_train.index)]   #including all labels (noise labels included)
df_test = df_all_test[df_all_test['primary_microconstituent'].isin (labels)]    #getting test excluding the noise labels

# save the tratin and test data
df_train.to_csv(file_path + 'train_data.csv', index=False)  # 4 labels
df_test.to_csv(file_path + 'test_data.csv', index=False)    # 4 labels
df_all_test.to_csv(file_path + 'all_test_data.csv', index=False)    # total 7 labels



print(all_labels)
print('len(df_train)= ', len(df_train))
print('len(df_test)= ', len(df_test))
print('len(df_all_test)= ', len(df_all_test))
print ('all_labels= ', all_labels)
print( 'test_labels=',df_test.primary_microconstituent.unique())  #to ensure the test data not include nose labels
'''
