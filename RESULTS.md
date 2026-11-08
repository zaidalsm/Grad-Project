# Results: first corrosion segmentation model

Model: U-Net with a pretrained ResNet34 encoder (segmentation-models-pytorch).
Trained on images resized to 512x512, 75 training images. The best epoch (38)
was chosen on validation IoU. Settings and library versions are in
runs/unet_resnet34/run_config.json.

## Scores

| Set | Images | IoU | Dice | Precision | Recall | Time per image |
|---|---|---|---|---|---|---|
| Validation | 15 | 0.8697 | 0.9303 | 0.9356 | 0.9251 | 0.088 s |
| Test | 15 | 0.8077 | 0.8936 | 0.8628 | 0.9267 | 0.109 s |

IoU here adds up all pixels from all images. Averaged per image instead, IoU
is 0.8276 (validation) and 0.8249 (test). Time was measured on a GPU and
includes reading the 2048x2048 file and drawing the overlay.

The model finds almost all the rust (recall 0.93) but also marks some areas
that are not rust (precision 0.86 on test). Validation is slightly higher
because the best epoch was picked on it.

## Worst cases on the test set

| 07.jpg (key on a wooden board) | 0.2604 | Real rust is 8.1% of the image but the model marks 23.7%: brown wood is marked as rust. The light round head of the key is missed. |
| 14.jpg (rusty nut on concrete) | 0.5244 | Real rust is only 2.2% of the image, the model marks 4.1%. The nut is found, but a false patch near a small white mark halves the score because the rust area is tiny. |
| 04.jpg (rusty plate with two rods) | 0.6892 | Real rust 64.0%, predicted 67.8%. The right-hand dark rod is missed and extra areas are marked (precision 0.79, recall 0.84). The mask edge is rough here. |

Pattern: the errors are mostly false alarms on rust-coloured surfaces (brown
wood, marks on concrete), plus rust that is darker than usual being missed.
This matches precision being lower than recall,the model over-marks rust on 07, 14 and 03 (03: true 38.9%, predicted 45.2%) and under-marks it on 08 (precision 0.97 but recall 0.79, true 22.0%, predicted 17.9%), so errors go in both directions.

## Limits of these results

- The test set has only 15 images, so the scores are rough.
- The dataset photos are mostly objects and surfaces, not the inside of a tank.
  These scores do not tell us how the model will do in the tank.
- Speed was measured on a GPU only. The Raspberry Pi is unknown.
- Only the 3 worst test images have been examined so far.

## Proposed success limit (to agree with the teacher)

Dice of at least 0.85 on images not used for training or for choosing
settings. This is fixed before the final test, which will use new photos.

## Training

Training ran for 40 epochs at about 2 s each (about 75 s in total on the GPU).
Training loss fell from 1.10 to about 0.2 and validation loss from 2.64 to
about 0.25. The two stay close, so the model is not overfitting. Validation
IoU rose from 0.51 to about 0.85-0.87. It swung a lot in early epochs (for
example 0.59 at epoch 5) because the validation set has only 15 images, and
it settled after epoch 35. The best epoch (38, validation IoU 0.869) is
within noise of epochs 36 to 40.

## Next (Week 5)

1. Look at the worst validation images and check for the same pattern.
2. Try colour and brightness changes in the training augmentation. Judge each
   change on validation scores only, never on test.
3. Test on phone photos of real rust under different lighting.
4. Record every experiment and its validation scores.