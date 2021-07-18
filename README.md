# best-comeback

Want to end an argument with a win? This repo uses facial landmarks to automatically generate a "Deal With It" gif from an input image.

## Development

**Install all packages in _requirements.txt_.**
**In order to install dlib, there are some prerequisites. Follow [this guide](https://www.pyimagesearch.com/2017/03/27/how-to-install-dlib/) which shows the setup process for dlib**

```
pip3 install -r requirements.txt
```

**Run with python**

```
python3 best_comeback.py <path/to/image>
```

**Or use a directory to run on all images in directory**

```
python3 best_comeback.py <path/to/directory>
```

## Examples

More [examples](./examples/).

<table style="table-layout: fixed; width: 100%;">
<tr>
  <td>Input</td>
  <td>Output</td>
</tr>
<tr>
  <td width="50%"><img src="examples/1.jpeg" /></td>
  <td width="50%"><img src="examples/1.gif" /></td>
</tr>
<tr>
  <td width="50%"><img src="examples/2.jpeg" /></td>
  <td width="50%"><img src="examples/2.gif" /></td>
</tr>
<tr>
  <td width="50%"><img src="examples/3.jpeg" /></td>
  <td width="50%"><img src="examples/3.gif" /></td>
</tr>
</table>
