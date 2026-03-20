# NorgesGruppen Data: Object Detection

Detect grocery products on store shelves. Upload your model code as a `.zip` file — it runs in a sandboxed Docker container on our servers.

## How It Works

1.  Download the training data from the competition website (requires login)
2.  Train your object detection model locally
3.  Write a `run.py` that takes shelf images as input and outputs predictions
4.  Zip your code + model weights
5.  Upload at the submit page
6.  Our server runs your code in a sandbox with GPU (NVIDIA L4, 24 GB VRAM) — no network access
7.  Your predictions are scored: **70% detection** (did you find products?) + **30% classification** (did you identify the right product?)
8.  Score appears on the leaderboard

## Downloads

Download training data and product reference images from the **Submit** page on the competition website (login required).

## Training Data

Two files are available for download:

**COCO Dataset** (`NM_NGD_coco_dataset.zip`, ~864 MB)

- 248 shelf images from Norwegian grocery stores
- ~22,700 COCO-format bounding box annotations
- 356 product categories (category_id 0-355) — detect and identify grocery products
- Images from 4 store sections: Egg, Frokost, Knekkebrod, Varmedrikker

**Product Reference Images** (`NM_NGD_product_images.zip`, ~60 MB)

- 327 individual products with multi-angle photos (main, front, back, left, right, top, bottom)
- Organized by barcode: `{product_code}/main.jpg`, `{product_code}/front.jpg`, etc.
- Includes `metadata.json` with product names and annotation counts

### Annotation Format

The COCO annotations file (`annotations.json`) contains:

<figure data-rehype-pretty-code-figure="">
<pre style="background-color:#0d1117;color:#e6edf3" tabindex="0" data-language="json" data-theme="github-dark-default"><code>{
  &quot;images&quot;: [
    {&quot;id&quot;: 1, &quot;file_name&quot;: &quot;img_00001.jpg&quot;, &quot;width&quot;: 2000, &quot;height&quot;: 1500}
  ],
  &quot;categories&quot;: [
    {&quot;id&quot;: 0, &quot;name&quot;: &quot;VESTLANDSLEFSA TØRRE 10STK 360G&quot;, &quot;supercategory&quot;: &quot;product&quot;},
    {&quot;id&quot;: 1, &quot;name&quot;: &quot;COFFEE MATE 180G NESTLE&quot;, &quot;supercategory&quot;: &quot;product&quot;},
    ...
    {&quot;id&quot;: 356, &quot;name&quot;: &quot;unknown_product&quot;, &quot;supercategory&quot;: &quot;product&quot;}
  ],
  &quot;annotations&quot;: [
    {
      &quot;id&quot;: 1,
      &quot;image_id&quot;: 1,
      &quot;category_id&quot;: 42,
      &quot;bbox&quot;: [141, 49, 169, 152],
      &quot;area&quot;: 25688,
      &quot;iscrowd&quot;: 0,
      &quot;product_code&quot;: &quot;8445291513365&quot;,
      &quot;product_name&quot;: &quot;NESCAFE VANILLA LATTE 136G NESTLE&quot;,
      &quot;corrected&quot;: true
    }
  ]
}</code></pre>
</figure>

Key fields: `bbox` is `[x, y, width, height]` in pixels (COCO format). `product_code` is the barcode. `corrected` indicates manually verified annotations.

## What Annotations Look Like

A training image with all ground truth boxes (green = correctly detected product):

<img src="/docs/shelf-annotations-full.png" style="max-width:100%; border-radius:8px; margin:8px 0" alt="Shelf image with all 76 products annotated in green bounding boxes — this represents a perfect mAP of 1.0" />

Compare with a ~50% mAP result — half the products are missed entirely, and some detected boxes (red) are imprecise:

<img src="/docs/shelf-annotations-partial.png" style="max-width:100%; border-radius:8px; margin:8px 0" alt="Same shelf image with only half the products detected, some with imprecise boxes shown in red — approximately 50% mAP" />

## Submit

Upload your `.zip` at the submission page on the competition website.

## MCP Setup

Connect this docs server to your AI coding tool:

<figure data-rehype-pretty-code-figure="">
<pre style="background-color:#0d1117;color:#e6edf3" tabindex="0" data-language="bash" data-theme="github-dark-default"><code>claude mcp add --transport http nmiai https://mcp-docs.ainm.no/mcp</code></pre>
</figure>
