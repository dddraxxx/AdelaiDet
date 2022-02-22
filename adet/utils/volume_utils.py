import SimpleITK as sitk

def read_niigz(path):
    image = sitk.ReadImage(path)
    data = sitk.GetArrayFromImage(image)
    data = data[None]
    return data