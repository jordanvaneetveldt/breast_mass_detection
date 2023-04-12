# Breast cancer detection in Python 

More info: https://medium.com/towards-data-science/end-to-end-breast-cancer-detection-in-python-part-1-13a1695d455 (1)

To execute 'create_dataset.py' you should have access to the INbreast dataset that can be requested here: http://medicalresearch.inescporto.pt/breastresearch/index.php/Get_INbreast_Database.

Then you should have the following structure to properly create the dataset:

```.
├── INbreast Release 1.0
│   ├── AllDICOMs
│   │   ├── 20586908_6c613a14b80a8591_MG_R_CC_ANON.dcm
│   │   ├── ...
│   │   └── ...
│   ├── AllXML
│   │   ├── 20586908.xml
│   │   ├── ...
└── └── └── ...
```

If you want to use another folder, you just need to modify DCM_PATH and XML_PATH in line 14 and 15 from ```create_dataset.py```. This will create a train, validation and test folder with the same procedure as described in my article (1). You can change the seed in line 19 to create a different dataset. 

Finally to reproduce my results, you can train YOLOv4 with my config file: ```yolov4-breast.cfg```.
