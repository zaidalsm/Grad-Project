    # Interface agreement: AI model <-> rest of the system

    Draft for the Week 2 team meeting. Change anything the team decides differently.

    ## Model input
    - One image file (PNG or JPG), RGB camera image, any size (the code resizes it).
    - Passed to the function as a file path.

    ## Model output
    - `mask`: single-channel PNG, same size as the input, values 0 (background) and 255 (corrosion)
    - `overlay`: the input image with the corrosion area tinted red
    - `corrosion_ratio`: number between 0 and 1 = share of image pixels marked as corrosion

    ## How other code calls it
    from predict_one import analyze_image
    result = analyze_image("inspection.jpg")
    result["mask"]             # numpy array, uint8, 0/255
    result["overlay"]          # numpy array, BGR (OpenCV)
    result["corrosion_ratio"]  # float, e.g. 0.12

    ## Files and names
    - Model weights: runs/unet_resnet34/best_model.pt
    - Inspection images: inspections/images/INSP-0001.jpg, log: inspections/log.csv
    - Predictions: predictions/<image name>_mask.png and _overlay.png

    ## What the result means (and does not)
    - The ratio is the visible corrosion area in ONE image, from one viewpoint.
    - It does not measure corrosion depth, thickness loss or remaining life.
    - Results on simulation images do not prove performance on a real tank.