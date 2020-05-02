import tensorflow as tf
import tensorflow_hub as hub
#https://www.tensorflow.org/tutorials/generative/style_transfer
import IPython.display as display
import matplotlib.pyplot as plt
import matplotlib as mpl
import PIL.Image
mpl.rcParams['figure.figsize'] = (12,12)
mpl.rcParams['axes.grid'] = False
import numpy as np


def tensor_to_image(tensor):
  tensor = tensor*255
  tensor = np.array(tensor, dtype=np.uint8)
  if np.ndim(tensor)>3:
    assert tensor.shape[0] == 1
    tensor = tensor[0]
  return PIL.Image.fromarray(tensor)

def load_img(path_to_img):
  max_dim = 512
  img = tf.io.read_file(path_to_img)
  img = tf.image.decode_image(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.float32)

  shape = tf.cast(tf.shape(img)[:-1], tf.float32)
  long_dim = max(shape)
  scale = max_dim / long_dim

  new_shape = tf.cast(shape * scale, tf.int32)

  img = tf.image.resize(img, new_shape)
  img = img[tf.newaxis, :]
  return img


def imshow(image, title=None):
  if len(image.shape) > 3:
    image = tf.squeeze(image, axis=0)

  plt.imshow(image)
  if title:
    plt.title(title)


#style_path = tf.keras.utils.get_file('kandinsky5.jpg','https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg')
#style_path = tf.keras.utils.get_file('joker.jpg','https://i.pinimg.com/originals/45/85/f8/4585f8ab7fd2ad7556f11b4d399f5445.jpg')
#style_path = tf.keras.utils.get_file('cel1.jpg','https://images-wixmp-ed30a86b8c4ca887773594c2.wixmp.com/f/33496ace-f80c-4e99-b3d1-a518c1c40ae8/d9cds6-4a7ffec0-c7c0-4a5b-bd97-927572ba6aa4.jpg/v1/fill/w_576,h_576,q_75,strp/cel_shaded_self_portrait_by_baka_baka_d9cds6-fullview.jpg?token=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ1cm46YXBwOjdlMGQxODg5ODIyNjQzNzNhNWYwZDQxNWVhMGQyNmUwIiwiaXNzIjoidXJuOmFwcDo3ZTBkMTg4OTgyMjY0MzczYTVmMGQ0MTVlYTBkMjZlMCIsIm9iaiI6W1t7ImhlaWdodCI6Ijw9NTc2IiwicGF0aCI6IlwvZlwvMzM0OTZhY2UtZjgwYy00ZTk5LWIzZDEtYTUxOGMxYzQwYWU4XC9kOWNkczYtNGE3ZmZlYzAtYzdjMC00YTViLWJkOTctOTI3NTcyYmE2YWE0LmpwZyIsIndpZHRoIjoiPD01NzYifV1dLCJhdWQiOlsidXJuOnNlcnZpY2U6aW1hZ2Uub3BlcmF0aW9ucyJdfQ.rSw9sI2-i9xrUPI0wq1duuh-AjSUqroKzknssG9JI1U')
style_path = tf.keras.utils.get_file('hest5.jpg','https://scontent-cph2-1.xx.fbcdn.net/v/t1.15752-9/95353440_258276062208684_1809513706564878336_n.jpg?_nc_cat=106&_nc_sid=b96e70&_nc_ohc=Iwa85hO3FQQAX_KCP0Z&_nc_ht=scontent-cph2-1.xx&oh=7d0a67dbc8128a6930cc4ceffe03be37&oe=5ED1D408')

#content_path = tf.keras.utils.get_file('YellowLabradorLooking_new.jpg', 'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg')
#content_path = tf.keras.utils.get_file('WilliamOgViktor.jpg', 'https://lh3.googleusercontent.com/R6Bqb_CB-6-_r5d0Z_r4TUDr5XtXzqupNBe0JgSD6Npu88W_GftxUrXBxDry65J0kjv9S2B9AVbzPDnzo8WGU7gEEN_QLt65tyLWMJ7snbxdyaYLWS1kjwlZ9OkHXSrKt5ZKiaFE0JL7BOKTpML9sDVAdZ0GBkIFTwAkSNp2hwbPGR9mdo6VeGr0Q9ExOSfzGXgGt6OMHhrHSW4BWZykU07XFcqFupp-Jk2JzwkqwxhFgeJe8BJMwMKD6nxRxJvhQM3paN8z1sP2aLTJlgsO8i5xXZVJ9ZSytJlvNggUUv6fEq_MLNFz64MRPh4C_nTF4R3b6vRXbx4_ieyYRrBFuJER5DPr_dKJDpK-t0vK3Ca6pXdGxUTuWWJ0Dwm6d2866Puav-px36eaEdnTPb_cSq1jmGhjQ_X4s-CFwbdYBzZNfNhcXhr4fCPKu_Q6g_DeCY36x6I6SvTTD0MD04nVTRe5NhrBkqYHISJPyfnskAPQBRzlSt53-sBDbB9tBFA5tbuQL4F7MPewqX5nbvgWDlIrR-ybH7ZHohGTdnWTwTOVIrX9QUF4FtjGkCgekkG8a7WM4nHrIlhP5vCoZN6ZdzoCbPivy5XMCuyYLGfkR5OGC6yBefu3_DRlBBc492vys9U2s6Xp1K7m-BcnVJc0y_8HuJUcxTQ6Gxr-zPZhruNSymK7dmK0tRW1X5StwUs=w1280-h960-no')
#content_path = tf.keras.utils.get_file('SOH1.jpg', 'https://scontent-cph2-1.xx.fbcdn.net/v/t1.0-9/222325_10150177429233402_492124_n.jpg?_nc_cat=108&_nc_sid=85a577&_nc_ohc=dOOooiT1iREAX_6uSaZ&_nc_ht=scontent-cph2-1.xx&oh=64951c74720338ff66a4ad2f89a71f1f&oe=5ED2B3DA')
content_path = tf.keras.utils.get_file('horsec12.jpg', 'https://c1.staticflickr.com/8/7436/9854312874_c0130b5a57_b.jpg')

content_img = load_img(content_path)
plt.subplot(221)
imshow(content_img, 'Content')

plt.subplot(222)
style_img = load_img(style_path)
imshow(style_img, 'Style')


hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
stylized_image = hub_module(tf.constant(content_img), tf.constant(style_img))[0]

plt.subplot(212)
res = tensor_to_image(stylized_image)
imshow(stylized_image, 'Final')

plt.show()

