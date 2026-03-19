# External Datasets for Object Detection

## Tier 1 (Must Use)

### SKU-110K - Dense Retail Shelf Detection
- **URL**: https://github.com/eg4000/SKU110K_CVPR19
- **Ultralytics YOLO-ready**: https://docs.ultralytics.com/datasets/detect/sku-110k/
- **Size**: ~11,762 images, ~1.73M bounding box annotations
- **Format**: YOLO-format via Ultralytics. Single class ("object").
- **License**: Academic/non-commercial
- **Content**: Dense shelf images, avg ~147 objects/image
- **Use**: Pre-training for dense shelf detection

### SKUs on Shelves PL (Polish Stores)
- **URL**: https://huggingface.co/datasets/shelfwise-by-form/SKUs_on_shelves_PL
- **Size**: ~27,000 images, ~2M annotations, ~8,000 unique SKUs
- **Format**: COCO format
- **License**: CC-BY-4.0
- **Content**: Real Polish retail store shelf images
- **Use**: Large-scale shelf detection training

## Tier 2 (Strong Supplements)

### Grocery Store Dataset (Swedish, KTH)
- **URL**: https://github.com/marcusklasson/GroceryStoreDataset
- **Size**: 5,125 images, 81 classes
- **License**: MIT
- **Content**: Swedish grocery stores, Nordic brands (Arla, Oatly)
- **Use**: Classification (no bboxes)

### Supermarket Shelves Dataset
- **URL**: https://www.kaggle.com/datasets/humansintheloop/supermarket-shelves-dataset
- **Size**: 45 images, 11,743 bboxes
- **License**: CC0 Public Domain
- **Use**: Dense shelf annotations

### HoloSelecta (European Vending)
- **URL**: https://data.mendeley.com/datasets/gz39ggf35n/1
- **Size**: 295 images, 10,035 instances, 109 classes
- **Format**: Pascal VOC, GTIN-labeled
- **License**: CC BY 4.0
