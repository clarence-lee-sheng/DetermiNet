#!/usr/bin/env python
# coding: utf-8
# from pycocotools.coco import COCO
import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import pickle
from evaluation.eval_det import generate_corrected_gt_json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

tf.config.set_visible_devices([], 'GPU')

parser = argparse.ArgumentParser(description='Training oracle model')

parser.add_argument("--image_dir", help="path to image dir", default="data/")
parser.add_argument("--tfrecords_dir", help="path to tfrecord annotations", default="annotations/tfrecords/oracle_tfrecords")
parser.add_argument("--checkpoint_dir", help="path to save checkpoints", default="'./oracle/oracle_modelweights/oracle_best_chkpt_030823'")
args = parser.parse_args()

def saveload(opt, name, variblelist):
    name = name + '.pickle'
    if opt == 'save':
        with open(name, 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump(variblelist, f)
            print('Data Saved')
            f.close()

    if opt == 'load':
        with open(name, 'rb') as f:  # Python 3: open(..., 'rb')
            var = pickle.load(f)
            print('Data Loaded')
            f.close()
        return var

# # Determiners Dataset baseline model

image_dir = args.image_dir
tfrecords_dir = args.tfrecords_dir
AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 64

ignore_order = tf.data.Options()
ignore_order.experimental_deterministic = False

# #### read and parse tfrecords


determiners = ["a", "an", "all", "any", "every", "my", "your", "this", "that", "these", "those", "some", "many",
               "few", "both", "neither", "little", "much", "either", "our", "no", "several", "half", "each", "the"]


def parse_tfrecord_fn(example, img_folder=image_dir):
    feature_description = {
        "file_name": tf.io.FixedLenFeature([], tf.string),
        #         "image": tf.io.FixedLenFeature([], tf.string),
        "image_id": tf.io.FixedLenFeature([], tf.int64),
        "caption": tf.io.VarLenFeature(tf.string),
        "caption_one_hot": tf.io.VarLenFeature(tf.int64),
        "areas": tf.io.VarLenFeature(tf.int64),
        "category_ids": tf.io.VarLenFeature(tf.int64),
        "output_category_ids": tf.io.VarLenFeature(tf.int64),
        "output_areas": tf.io.VarLenFeature(tf.int64)
    }

    sequence_features = {
        "input_bboxes": tf.io.VarLenFeature(tf.float32),
        "output_bboxes": tf.io.VarLenFeature(tf.float32),
        "input_one_hot": tf.io.VarLenFeature(tf.float32),
        "output_one_hot": tf.io.VarLenFeature(tf.float32)
    }
    context, sequence = tf.io.parse_single_sequence_example(example, context_features=feature_description,
                                                            sequence_features=sequence_features)

    example = {**context, **sequence}
    for key in example.keys():
        if type(example[key]) == tf.sparse.SparseTensor:
            if (example[key].dtype == "string"):
                example[key] = tf.sparse.to_dense(example[key], default_value='b')
            else:
                example[key] = tf.sparse.to_dense(example[key])

    #     example["image"] = tf.io.decode_png(example["image"], channels=3)
    print(image_dir+example["file_name"])
    raw = tf.io.read_file(image_dir+example["file_name"])
    example["image"] = tf.io.decode_png(raw, channels=3)
    #     image = example["image"]
    #     print(example["caption_one_hot"])

    return example


def map_to_inputs(example):
    image = example["image"]
    caption = example["caption"]
    input_bbox = example["input_bboxes"]
    input_label = example["category_ids"]
    output_labels = example["output_category_ids"]
    input_one_hot = tf.cast(example["input_one_hot"], dtype=tf.float32)
    output_bboxes = example["output_bboxes"]
    output_one_hot = tf.cast(example["output_one_hot"], dtype=tf.float32)
    caption_one_hot = example["caption_one_hot"]

    input_one_hot = tf.concat([input_one_hot[:, :4] / 256, input_one_hot[:, 4:]], axis=1)
    output_one_hot = tf.concat([output_one_hot[:, :4] / 256, output_one_hot[:, 4:]], axis=1)
    input_mask = tf.stack(input_one_hot[:, 4:])
    output_mask = tf.stack(output_one_hot[:, 4])
    n_pad = 20 - tf.shape(output_labels)[0]
    output_labels_padded = tf.pad(output_labels, [[0, n_pad]], "CONSTANT")
    return (input_mask, caption_one_hot), (output_mask)


train_filenames = tf.io.gfile.glob(f"{tfrecords_dir}/train/*.tfrec")
print("train filenames: ", train_filenames)

train_dataset = tf.data.TFRecordDataset(train_filenames, num_parallel_reads=AUTOTUNE)
train_dataset = train_dataset.map(parse_tfrecord_fn, num_parallel_calls=AUTOTUNE)
train_dataset = train_dataset.map(map_to_inputs, num_parallel_calls=AUTOTUNE)
train_dataset = train_dataset.shuffle(4 * BATCH_SIZE)
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
train_dataset = train_dataset.batch(BATCH_SIZE)
example = next(iter(train_dataset))

val_filenames = tf.io.gfile.glob(f"{tfrecords_dir}/val/*.tfrec")
print(val_filenames)
val_dataset = tf.data.TFRecordDataset(val_filenames, num_parallel_reads=AUTOTUNE)
val_dataset = val_dataset.map(parse_tfrecord_fn, num_parallel_calls=AUTOTUNE)
val_dataset = val_dataset.map(map_to_inputs, num_parallel_calls=AUTOTUNE)
val_dataset = val_dataset.prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.batch(BATCH_SIZE)

#print(example[0][1])

# neural bounding box selector model architecture

nemb = 128
nhid = 256
maxbb = 20
nclass = 25
atype = 'none'

class SimpleBboxSelector(tf.keras.Model):
    def __init__(self):
        super(SimpleBboxSelector, self).__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.img_embed = tf.keras.layers.Dense(nemb, activation='relu', name='img_emb')  # same shape as bb+lbl
        self.cap_embed = tf.keras.layers.Dense(nemb,activation='relu', name='cap_emb')  # same shape as bb+lbl

        self.dense1 = tf.keras.layers.Dense(nhid,activation='relu', name='combine')
        self.dense2 = tf.keras.layers.Dense(nhid, activation='relu')
        self.bb_mask = tf.keras.layers.Dense(maxbb,activation='sigmoid',name='bb_mask')
        #self.det_class = tf.keras.layers.Dense(nclass, activation='softmax', name='det_class')

    def call(self,inputs):
        mask = np.ones_like(inputs[1])
        #mask[:,:] = 0  # determine which captions to omit, determiners [:,:25], nouns [:,25:]
        if atype == 'ydnn':
            mask[:, 25:] = 0
        elif atype == 'ndyn':
            mask[:, :25] = 0
        elif atype == 'ndnn':
            mask[:, :] = 0
        else:
            pass # no ablation of caption
        masked_cap = tf.multiply(inputs[1], mask)

        bb = self.img_embed(tf.cast(self.flatten(inputs[0]),dtype=tf.float32))
        c = self.cap_embed(tf.cast(masked_cap,dtype=tf.float32))
        bb_c = tf.concat([bb,c],axis=1) # concat/multiply/add
        x = self.dense2(self.dense1(bb_c))
        bbs = self.bb_mask(x)
        #det = self.det_class(x)
        return bbs


optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
loss_fn = tf.keras.losses.BinaryCrossentropy()
model = SimpleBboxSelector()
model.compile(optimizer=optimizer, loss=loss_fn, metrics=['binary_crossentropy'],run_eagerly=True)

# train model on dataset
epochs = 30

checkpoint_filepath = args.checkpoint_dir
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath, save_weights_only=True, monitor='val_loss', mode='min', save_best_only=True)

# print(tf.config.list_physical_devices('GPU'))
#with tf.device('/device:GPU:0'):
history = model.fit(train_dataset, validation_data=val_dataset, epochs=epochs,callbacks=[model_checkpoint_callback])
tr_loss = history.history['loss']
val_loss = history.history['val_loss']
print('TR loss: {}'.format(tr_loss))
print('Val loss: {}'.format(val_loss))

model.save_weights("./oracle_modelweights/oracle_weights_e{}_030823.h5".format(epochs))

# model.predict([np.zeros([1,440]),np.zeros([1,41])])
model.load_weights("./oracle_modelweights/oracle_best_chkpt_030823")
print(model.summary())
# load & evaluate model on test datset

# test_filenames = tf.io.gfile.glob(f"{tfrecords_dir}/test/*.tfrec")
# print(test_filenames)
# test_dataset = tf.data.TFRecordDataset(test_filenames, num_parallel_reads=AUTOTUNE)
# test_dataset = test_dataset.map(parse_tfrecord_fn, num_parallel_calls=AUTOTUNE)
# test_dataset = test_dataset.map(map_to_inputs, num_parallel_calls=AUTOTUNE)
# test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)
# test_dataset = test_dataset.batch(BATCH_SIZE)

# model.evaluate(test_dataset)
# preds = model.predict(test_dataset)
# print(preds[1].shape)


test_filenames = tf.io.gfile.glob(f"{tfrecords_dir}/test/*.tfrec")
print(test_filenames)
test_dataset = tf.data.TFRecordDataset(test_filenames, num_parallel_reads=AUTOTUNE)
examples = test_dataset.map(parse_tfrecord_fn, num_parallel_calls=AUTOTUNE)

results = []
#results.append({"loss": tr_loss, 'valloss': val_loss})

pred_bb = []
pred_cls = []
pred_score = []
gt_bb = []
gt_cls = []
all_bb = []
all_emb = []

for i, example in enumerate(examples):

    if i%5000==0:
        print(i)
        print(example['file_name'])
        print(example['image_id'])
        print(example['caption'])
        print(np.argmax(example["caption_one_hot"][:25])) # det
        print(np.argmax(example["caption_one_hot"][25:]))  # obj
        print(np.argmax(example["input_one_hot"][:,5:],axis=1))  # det
        print(example["input_one_hot"][:,-1])  # det
        print(example["output_bboxes"][:,:4])

    inputs, outputs = map_to_inputs(example)

    inputs = tf.expand_dims(inputs[0], axis=0), tf.expand_dims(inputs[1], axis=0)
    pred_tr_score = model(inputs)
    pred_tr_score = pred_tr_score.numpy()
    pred_ts_bb = (example["input_one_hot"].numpy()[:, :4] * (pred_tr_score > 0.5)[:, :, None])

    category_id = 1

    for idx in np.arange(20)[pred_tr_score[0] > 1e-10]:
        bbox = pred_ts_bb[0][idx]
        results.append(
            {"image_id": int(example["image_id"].numpy()), "bbox": bbox.tolist(), "category_id": int(category_id),
             "score": float(pred_tr_score[0][idx])})

json.dump(results, open(os.path.join("oracle_results/oracle_e{}_test_results_030823.json".format(epochs)), "w"))


# correct ground truth and evaluate

annFile = './DetermiNetV2/annotations_test.json'
cocoGt = COCO(annFile)

resFile = './oracle_results/oracle_e{}_test_results_030823.json'.format(epochs)
cocoDt = cocoGt.loadRes(resFile)

annType = "bbox"

# cocoEval = COCOeval(cocoGt, cocoDt, annType)
# cocoEval.params.imgIds = sorted(cocoGt.getImgIds())
# cocoEval.evaluate()
# cocoEval.accumulate()
# cocoEval.summarize()

modannFile = './oracle_results/modgt_oracle_e{}_test_results_030823.json'.format(epochs)
generate_corrected_gt_json(gt_dir=annFile, results_dir=resFile,mod_res_dir=modannFile)
modcocoGt = COCO(modannFile)

print('After correcting annotations')
cocoEval = COCOeval(modcocoGt, cocoDt, annType)
cocoEval.params.imgIds = sorted(cocoGt.getImgIds())
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()

plt.figure()
plt.plot(tr_loss)
plt.plot(val_loss)
plt.title('TR:{:.2g}, Val:{:.2g}'.format(tr_loss[-1],val_loss[-1]))
plt.savefig("oracle_syn_e{}_b{}.png".format(epochs, BATCH_SIZE))
