import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,Dropout,BatchNormalization
from tensorflow.keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
import pickle
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

path=os.listdir("Face_Data")


train_path=os.listdir("Face_data"+"\\"+"train")

test_path=os.listdir("Face_Data"+"\\"+"test")


images=[]
target=[]


for i in train_path:
    myList=os.listdir("Face_Data"+"\\"+"train"+"\\"+str(i))
    for j in myList:
        img=cv2.imread("Face_Data"+"\\"+"train"+"\\"+str(i)+"\\"+str(j))
        img=cv2.resize(img,(48,48))
        images.append(img)
        target.append(i)
        
        

        
images=np.array(images)       
target=np.array(target)    

print(images.shape)
print(target.shape)
label_encoder = LabelEncoder()
target = label_encoder.fit_transform(target)

print(target[0])
#%%train test
x_train,x_test,y_train,y_test=train_test_split(images,target,test_size=0.19,random_state=2)

#x_train,x_val,y_train,y_val=train_test_split(x_train,y_train,test_size=0.2,random_state=42)

print("x shape:",x_test.shape)
print("y_shape: ",y_test.shape)
#print(x_val.shape)

#%% preprocessing
def preProcess(img):
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img=cv2.equalizeHist(img)
    img=img/255
    return img

x_train=np.array(list(map(preProcess,x_train)))#map kullandığımız fonksiyonu x_train içindeki her şey için uygular
x_test=np.array(list(map(preProcess,x_test)))        
#x_val=np.array(list(map(preProcess,x_val)))


   
x_train=x_train.reshape(-1,48,48,1)    

#%%datagenarator
'''
dataGen=ImageDataGenerator(width_shift_range=0.1,
                           height_shift_range=0.1,
                           zoom_range=0.1,
                           rotation_range=10
    )
'''
print("x shape:",x_test.shape)
print("y_shape: ",y_test.shape)

#%%model training evaulation
#dataGen.fit(x_train)



y_train=to_categorical(y_train,7)#7 seçeneğimiz olduğu için 7 farklı gruba ayıracağımız için 7 yazdım
y_test=to_categorical(y_test,7)
#y_val=to_categorical(y_val,7)
print("y train: ",y_train[0])

model=Sequential()

model.add(Conv2D(input_shape=(48,48,1),filters=8, kernel_size=(5,5), activation="relu",padding="same"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters=16, kernel_size=(3,3), activation="relu",padding="same"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(units=512,activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(units=7,activation="softmax"))
model.summary()
model.compile(loss="categorical_crossentropy",optimizer=("Adam"),metrics=["accuracy"])


'''
hist=model.fit_generator(model.flow(x_train,y_train,batch_size=batch_size),
                                      validation_data=(x_test,y_test),
                                      epochs=15,steps_per_epoch=x_train.shape[0]//batch_size,shuffle=1
                                      )
'''

'''
for i in batch_sizes:
    model.fit(x_train,y_train,epochs=5,batch_size=i)
    score_test=model.evaluate(x_test,y_test)
    print("test accuracy: ",score_test[1])
'''
model.fit(x_train,y_train,epochs=30,batch_size=128)#epoch datamızın kaç kere eğitileceğini gösteriyor
score_test=model.evaluate(x_test,y_test)
print("test accuracy: ",score_test[1])

y_pred=model.predict(x_test)
y_pred_class=np.argmax(y_pred,axis=(1))
y_truth=np.argmax(y_test,axis=1)


    
#print("prediction: ",y_pred_class[10])
#print("truth: ",y_truth[10])

def find_emotion(emt):
        if emt==0:
          return"angry"
        if emt==1:
            return"disgusted"
        if emt==2:
         return"fearful"
        if emt==3:
             return"happy"
        if emt==4:
             return "neutral"
        if emt==5:
           return"sad"
        if emt==6:
            return"suprised"
            
print("prediction: ",find_emotion(y_pred_class[10]))
print("truth: ",find_emotion(y_truth[10]))

plt.figure("figure")
plt.imshow(x_test[10],cmap="gray")
plt.axis("off")
