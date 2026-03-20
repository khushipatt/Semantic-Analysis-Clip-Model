#  Semantic Image Search — NLP Sem 6 Assignment

This is a semantic image search app I built using OpenAI's CLIP model. The idea is simple, instead of searching images by their filename or tags, you just type something like *"dog in hands"* or *"flowers in a car"* and it finds the most visually matching photo from your collection. No manual labeling, no metadata — just the image and a text query.

---

##  What's actually happening here?

Normal search looks for keywords. This doesn't do that at all.

CLIP (Contrastive Language–Image Pretraining) is a model trained on a massive amount of image-text pairs. What's cool about it is that it learned to understand both images and text in the *same* space — meaning it can compare a photo of a sloth holding a strawberry to the sentence "cute animal with fruit" and tell you they're similar.

So when you type a query, the app:
1. Converts your text into a vector (a list of numbers representing meaning)
2. Does the same for every image in your folder
3. Finds whichever image vector is closest to your text vector
4. Shows you the best match

That's it. No training required on your end — CLIP already knows what things look like.

---

##  Running it

**1. Install the packages**
```bash
pip install -r requirements.txt
```

**2. Add your photos**

Create a folder called `my_photos` in the project root and drop your images in there (`.jpg`, `.jpeg`, or `.png`).

```
my_photos/
  ├── night_flight.jpg
  ├── dachshund.jpg
  ├── flowers.jpg
  └── ...
```

**3. Start the app**
```bash
streamlit run app.py
```

**4. Open your browser** at `http://localhost:8501`, type a query, and see what it finds.

---

## What's in the repo

```
├── app.py              # The whole app — loads CLIP, processes images, handles search
├── requirements.txt    # Python packages you'll need
├── my_photos/          # Put your images here (not included in repo)
├── .devcontainer/
│   └── devcontainer.json   # GitHub Codespaces config (runs automatically)
└── .gitignore          # Standard Python ignores
```

---

##  Sample images I tested with

The app was tested with a mix of photos to make sure CLIP could tell them apart:

- A blue embroidered saree on a mannequin
- A baby sloth holding a strawberry
- A bouquet of pink roses and lilies
- A plane engine at night on the tarmac
- A dachshund puppy being held up

Queries like `"dog in hands"`, `"flowers"`, `"airplane at night"`, and `"cute animal"` all returned the right image. CLIP handles this kind of varied, everyday photo set really well.

---

##  How the code works

Everything lives in `app.py`. Here's what it does step by step:

**Loading the model**
It uses `open_clip` with the `RN50` model pretrained on `yfcc15m`. RN50 is a smaller, faster variant of CLIP — good enough for a college project and doesn't take forever to download.

**Processing the images**
On startup, it loops through every image in `my_photos/`, preprocesses each one (resize, normalize), and runs it through CLIP's image encoder to get an embedding. These embeddings get stored in memory.

**Handling a search query**
When you type something in the text box, it tokenizes your query and runs it through CLIP's text encoder. Then it computes the dot product similarity between your text embedding and every image embedding, and picks the one with the highest score.

**Showing the result**
It displays the filename and the image. Simple as that.

---

##  Libraries used

| Library | Why it's here |
|---|---|
| `open-clip-torch` | The CLIP model itself — used for both image and text encoding |
| `streamlit` | The web UI |
| `torch` | Running the model |
| `torchvision` | Image transforms and preprocessing |
| `Pillow` | Opening and converting image files |
| `numpy` | Computing dot product similarity between embeddings |
| `transformers` | Pulled in as a dependency, not used directly |

