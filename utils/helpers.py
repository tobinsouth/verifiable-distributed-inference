import numpy as np
import base64
import io


# Helper function to allow for debug prints to be turned on/off while keeping code readable
def conditional_print(message, verbose=True):
    if verbose:
        print(message)


def decode_b64_to_np_array(b64_input: bytes) -> np.ndarray:
    decoded_data: bytes = base64.b64decode(b64_input)
    byte_stream = io.BytesIO(decoded_data)
    np_arr = np.load(byte_stream)
    return np_arr


def encode_np_array_to_b64(np_array: np.ndarray) -> bytes:
    byte_stream = io.BytesIO()
    np.save(byte_stream, np_array)
    byte_stream.seek(0)
    encoded_np_arr = base64.b64encode(byte_stream.read())
    return encoded_np_arr


def rmse(y_pred, y_true):
    y_pred = y_pred.astype(np.float32).flatten()
    y_true = y_true.astype(np.float32).flatten()
    return np.sqrt(np.mean(np.square(y_true - y_pred)))


def rmspe(y_pred, y_true, epsilon=1e-10) -> float:
    y_pred = y_pred.astype(np.float32).flatten()
    y_true = y_true.astype(np.float32).flatten()

    errors = (y_true - y_pred) / (y_true + epsilon)
    return np.sqrt(np.mean(np.square(errors)))
