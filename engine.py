
"""Classification Engine used for classification tasks."""
from edgetpu.basic.python.basic_engine import BasicEngine
import numpy
from PIL import Image


class AutopilotEngine(BasicEngine):
  """Engine used for classification task."""

  def __init__(self, model_path):
    """Creates a BasicEngine with given model.

    Args:
      model_path: String, path to TF-Lite Flatbuffer file.

    Raises:
      ValueError: An error occurred when the output format of model is invalid.
    """
    BasicEngine.__init__(self, model_path)
    output_tensors_sizes = self.get_all_output_tensors_sizes()
    if output_tensors_sizes.size != 2:
      raise ValueError(
          ('Autopilot model should have 2 output tensors. Angle and Throttle'
           'This model has {}.'.format(output_tensors_sizes.size)))

  def ClassifyWithImage(
      self, img, resample=Image.NEAREST):
    """Classifies image with PIL image object.

    This interface assumes the loaded model is trained for image
    classification.

    Args:
      img: PIL image object.
      threshold: float, threshold to filter results.
      top_k: keep top k candidates if there are many candidates with score
        exceeds given threshold. By default we keep top 3.
      resample: An optional resampling filter on image resizing. By default it
        is PIL.Image.NEAREST. Complex filter such as PIL.Image.BICUBIC will
        bring extra latency, and slightly better accuracy.

    Returns:
      List of (int, float) which represents id and score.

    Raises:
      RuntimeError: when model isn't used for image classification.
    """
    input_tensor_shape = self.get_input_tensor_shape()
    print(f'input_tensor_shape {input_tensor_shape}')
    if (input_tensor_shape.size != 4 or input_tensor_shape[3] != 3 or
        input_tensor_shape[0] != 1):
      raise RuntimeError('Invalid input tensor shape! Expected : [1, width, height, 3]')
    required_image_size = (input_tensor_shape[1], input_tensor_shape[2])
    print(f'required_image_size {required_image_size}')
    img = img.resize(required_image_size, resample)
    input_tensor = numpy.asarray(img).flatten()
    print(f'input_tensor_shape {input_tensor.shape}')
    print(f'input_tensor {input_tensor}')
    return self.ClassifyWithInputTensor(input_tensor)

  def ClassifyWithInputTensor(self, input_tensor):
    """Classifies with raw input tensor.

    This interface requires user to process input data themselves and convert
    it to formatted input tensor.

    Args:
      input_tensor: numpy.array represents the input tensor.

    Returns:
      List of (int, float) which represents id and score.

    Raises:
      ValueError: when input param is invalid.
    """

    _, self._raw_result = self.RunInference(
        input_tensor)
    result = []
    print('raw_result_retrieved')
    return self._raw_result