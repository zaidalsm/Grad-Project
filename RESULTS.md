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

| Image | IoU | What went wrong |
|---|---|---|
| [file name] key on a wooden board | about 0.26 | The key is found, but large brown areas of the wood are also marked as rust (false alarm). The light round head of the key is missed. |
| [file name] rusty plate with two rods | 0.69 | The big rust area is found, but the right-hand rod (dark, purple-brown) is not marked, and some extra areas at the edges are. The hand-drawn mask is rough here, so part of the gap may come from the labels. |
| [file name] rusty nut on concrete | 0.52 | The nut is marked correctly, but a patch around a small white mark at the bottom is also marked. The rust area is small, so one false patch lowers the score a lot. |

Pattern: the errors are mostly false alarms on rust-coloured surfaces (brown
wood, marks on concrete), plus rust that is darker than usual being missed.
This matches precision being lower than recall.

## Limits of these results

- The test set has only 15 images, so the scores are rough.
- The dataset photos are mostly objects and surfaces, not the inside of a tank.
  These scores do not tell us how the model will do in the tank.
- Speed was measured on a GPU only. The Raspberry Pi is unknown.
- Only the 3 worst test images have been examined so far.

## Proposed success limit (to agree with the teacher)

Dice of at least 0.85 on images not used for training or for choosing
settings. This is fixed before the final test, which will use new photos.

## Next (Week 5)

1. Look at the worst validation images and check for the same pattern.
2. Try colour and brightness changes in the training augmentation. Judge each
   change on validation scores only, never on test.
3. Test on phone photos of real rust under different lighting.
4. Record every experiment and its validation scores.