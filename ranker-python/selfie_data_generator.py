import pandas as pd
import cv2
import numpy as np

# read labels
file_ = "../Selfie-dataset/selfie_dataset.txt"
df = pd.read_csv(file_, header=None, delim_whitespace=True)
data = df.ix[:,0:1]
data = np.array(data)
# seperate dataset
traing_ratio = 0.8
total_amounts = len(data)
trains = data[0:int(traing_ratio*total_amounts)]
vals = data[int(traing_ratio*total_amounts):total_amounts]
# print trains.shape
# print vals
def read_image(img_name, folder="/home/adwin/Desktop/selfi-cv/Selfie-dataset/images/"):
	image_URL = folder+img_name+".jpg"
	im = cv2.resize(cv2.imread(image_URL), (224, 224)).astype(np.float32)
	im[:,:,0] -= 103.939
	im[:,:,1] -= 116.779
	im[:,:,2] -= 123.68
	im = im.transpose((2,0,1))
	#im = np.expand_dims(im, axis=0)
	#im = im/255
	return im

def train_generator(train_amount=len(trains), batch=32):
    training_sets = train_amount/batch # the amount of batches
    print "[Info] Training data total amount: ", len(trains)   
    print "[Info] Training data using amount: ", train_amount
    print "[Info] Batch size : ", batch
    print "[Info] Total batchs: ", training_sets

    while 1:
        for i in range(int(training_sets)):
            input_image_batch = np.zeros((batch,3,224,224))
            output_score_batch = np.zeros(batch)

            for j in range(batch):
            	index_ = i*batch + j
            	input_image_batch[j] = read_image(img_name=trains[index_][0])
            	output_score_batch[j] = trains[index_][1]

            inputs = input_image_batch
            outputs = output_score_batch
            yield (inputs, outputs)

def validation_generator(val_amount=len(vals), batch=32):
    val_sets = val_amount/batch # the amount of batches
    print "[Info] Val data total amount: ", len(vals)   
    print "[Info] Val data using amount: ", val_amount
    print "[Info] Batch size : ", batch
    print "[Info] Total batchs: ", val_sets

    while 1:
		for i in range(int(val_sets)):
			input_image_batch = np.zeros((batch,3,224,224))
			output_score_batch = np.zeros(batch)
			for j in range(batch):
				index_ = i*batch + j
				input_image_batch[j] = read_image(img_name=vals[index_][0])
				output_score_batch[j] = vals[index_][1]
			inputs = input_image_batch
			outputs = output_score_batch
			yield (inputs, outputs)
def test(train_amount=len(trains), batch=32):
    training_sets = train_amount/batch # the amount of batches
    print "[Info] Training data total amount: ", len(trains)   
    print "[Info] Training data using amount: ", train_amount
    print "[Info] Batch size : ", batch
    print "[Info] Total batchs: ", training_sets

    for i in range(int(training_sets)):
        input_image_batch = np.zeros((batch,3,224,224))
        output_score_batch = np.zeros(batch)

        for j in range(batch):
    		index_ = i*batch + j
        	input_image_batch[j] = read_image(img_name=trains[index_][0])
        	output_score_batch[j] = trains[index_][1]

        inputs = input_image_batch
        outputs = output_score_batch
        yield (inputs, outputs)
if __name__ == '__main__':
	aa = test()
	print aa.next()