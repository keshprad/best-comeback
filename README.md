# best-comeback

Want to end an argument with a win? This repo uses facial landmarks to automatically generate a "Deal With It" gif.

## Development

**Install all packages in _requirements.txt_.**
**In order to install dlib, there are some prerequisites. Follow [this guide](https://www.pyimagesearch.com/2017/03/27/how-to-install-dlib/) which shows the setup process for dlib**

```
pip3 install -r requirements.txt
```

**Run with python**

```
python3 main.py <path/to/image>
```

**Or use a directory to run on all images in directory**

```
python3 main.py <path/to/directory>
```

## Examples

More [examples](./examples/).

| Input                       | Output                      |
| --------------------------- | --------------------------- |
| ![Shaq_in](examples/1.jpeg) | ![Shaq_out](examples/1.gif) |
| ![AG_in](examples/2.jpeg)   | ![AG_out](examples/2.gif)   |
| ![B99_in](examples/3.jpeg)  | ![B99_out](examples/3.gif)  |
