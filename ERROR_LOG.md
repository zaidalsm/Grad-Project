# Error log

| Image | What went wrong | Likely cause | Fix tried |
|---|---|---|---|
| test 07.jpg (key on wood) | Real rust 8.1% of the image, model marks 23.7%. Brown wood marked as rust. Light round key head missed. | Wood colour and texture look like rust. | none yet |
| test 14.jpg (nut on concrete) | Real rust only 2.2%, model marks 4.1%. False patch near a white mark. | Rust area is tiny, so one false patch halves the score. Detail may be lost when shrinking 2048 to 512. | none yet |
| test 04.jpg (plate with rods) | Right-hand dark rod missed, extra areas marked (precision 0.79, recall 0.84). | Dark rust looks unlike typical orange rust. The hand-drawn mask is rough here. | none yet |
| test 08.jpg | Under-marks rust: precision 0.97, recall 0.79. | [look at the image and fill in] | none yet |
| val worst cases | [one line each from the three val_worst images] | | |