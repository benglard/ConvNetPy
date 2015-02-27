# ConvNetPy
ConvNetPy is a Python port of the wonderful [ConvNetJS](https://github.com/karpathy/convnetjs) I wrote for fun and learning.

## Disclaimers
- Given my goals in writing ConvNetPy (fun and learning), some code may be incomplete (almost nothing), may not work (almost nothing), may be badly documented (occasionally), or may be written in bad style (judge for yourself).
- Python is rarely fast enough. ConvNetPy is pure-python so [PyPy](http://pypy.org/) will work so long as you are careful using outside libraries (SciPy, OpenCV, NLTK, etc.). /examples has some examples where I used PyPy to train a model and then did some visualizations with OpenCV in a different Python distribution. PyPy's performance impressed me overall but it is still impractical for large models.
- I have data and models folders in my local copy of ConvNetPy which were unfortunately too large to upload to GitHub (they are used in /examples). I suppose you can email me if you want some piece of data or model I have.

Thank you very much to the original author (Andrej Karpathy) and to all the contributors to ConvNetJS!
