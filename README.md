# Neural Part-of-Speech tagger examples

This repository contains the code for DyNet, PyTorch and Tensorflow versions of a reasonably good POS tagger.
It also contains a program to convert that code (and comments) into a website for easy reading and comparison.

For the website, go [here](http://jkk.name/neural-tagger-tutorial/).

To generate the site, run:

```
./visualise.py tagger.dy.py tagger.pt.py tagger.tf.py > docs/index.html
```

Todo:
- Check results still match / code has not been broken

Ideas:
- Maybe add additional comments as roll-over pop-ups (and some indication that they are there)
