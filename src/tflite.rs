// (c) Yasuhiro Fujii <http://mimosa-pudica.net>, under MIT License.
use std::ffi::c_void;
use std::*;

#[link(name = "tensorflowlite_c")]
extern "C" {
    fn TfLiteModelCreate(_: *const u8, _: usize) -> *mut c_void;
    fn TfLiteModelDelete(_: *mut c_void);
    fn TfLiteInterpreterCreate(_: *const c_void, _: *const c_void) -> *mut c_void;
    fn TfLiteInterpreterDelete(_: *mut c_void);
    fn TfLiteInterpreterAllocateTensors(_: *mut c_void) -> i32;
    fn TfLiteInterpreterGetInputTensorCount(_: *const c_void) -> i32;
    fn TfLiteInterpreterGetInputTensor(_: *const c_void, _: i32) -> *mut c_void;
    fn TfLiteInterpreterInvoke(_: *mut c_void) -> i32;
    fn TfLiteInterpreterGetOutputTensorCount(_: *const c_void) -> i32;
    fn TfLiteInterpreterGetOutputTensor(_: *const c_void, _: i32) -> *const c_void;
    fn TfLiteTensorType(_: *const c_void) -> i32;
    fn TfLiteTensorNumDims(_: *const c_void) -> i32;
    fn TfLiteTensorDim(_: *const c_void, _: i32) -> i32;
    fn TfLiteTensorByteSize(_: *const c_void) -> usize;
    fn TfLiteTensorData(_: *const c_void) -> *mut c_void;
}

pub struct Interpreter {
    c_obj: *mut c_void,
}

#[derive(Debug)]
pub struct TensorRef<'a> {
    _phantom: marker::PhantomData<&'a ()>,
    pub type_id: any::TypeId,
    pub dims: Vec<usize>,
    n_bytes: usize,
    data: *const c_void,
}

#[derive(Debug)]
pub struct TensorRefMut<'a> {
    _phantom: marker::PhantomData<&'a ()>,
    pub type_id: any::TypeId,
    pub dims: Vec<usize>,
    n_bytes: usize,
    data: *mut c_void,
}

impl Drop for Interpreter {
    fn drop(&mut self) {
        unsafe {
            TfLiteInterpreterDelete(self.c_obj);
        }
    }
}

impl Interpreter {
    pub fn new(data: &[u8]) -> Self {
        let c_obj;
        unsafe {
            let model = TfLiteModelCreate(data.as_ptr(), data.len());
            assert_ne!(model, ptr::null_mut());
            c_obj = TfLiteInterpreterCreate(model, ptr::null());
            assert_ne!(c_obj, ptr::null_mut());
            TfLiteModelDelete(model);
            let status = TfLiteInterpreterAllocateTensors(c_obj);
            assert_eq!(status, 0);
        }

        Interpreter { c_obj: c_obj }
    }

    pub fn inputs<'a>(&'a self) -> Vec<TensorRefMut<'a>> {
        unsafe {
            (0..TfLiteInterpreterGetInputTensorCount(self.c_obj))
                .map(|i| {
                    let t = TfLiteInterpreterGetInputTensor(self.c_obj, i);
                    assert_ne!(t, ptr::null_mut());
                    TensorRefMut {
                        _phantom: marker::PhantomData,
                        type_id: Self::tensor_type_id(t),
                        dims: Self::tensor_dims(t),
                        n_bytes: TfLiteTensorByteSize(t),
                        data: TfLiteTensorData(t),
                    }
                })
                .collect()
        }
    }

    pub fn outputs<'a>(&'a self) -> Vec<TensorRef<'a>> {
        unsafe {
            (0..TfLiteInterpreterGetOutputTensorCount(self.c_obj))
                .map(|i| {
                    let t = TfLiteInterpreterGetOutputTensor(self.c_obj, i);
                    assert_ne!(t, ptr::null());
                    TensorRef {
                        _phantom: marker::PhantomData,
                        type_id: Self::tensor_type_id(t),
                        dims: Self::tensor_dims(t),
                        n_bytes: TfLiteTensorByteSize(t),
                        data: TfLiteTensorData(t),
                    }
                })
                .collect()
        }
    }

    pub fn invoke(&self) {
        unsafe {
            let status = TfLiteInterpreterInvoke(self.c_obj);
            assert_eq!(status, 0);
        }
    }

    fn tensor_dims(t: *const c_void) -> Vec<usize> {
        let n = unsafe { TfLiteTensorNumDims(t) };
        (0..n).map(|i| unsafe { TfLiteTensorDim(t, i) as usize }).collect()
    }

    fn tensor_type_id(t: *const c_void) -> any::TypeId {
        let i = unsafe { TfLiteTensorType(t) };
        match i {
            0 => any::TypeId::of::<()>(),
            1 => any::TypeId::of::<f32>(),
            2 => any::TypeId::of::<i32>(),
            3 => any::TypeId::of::<u8>(),
            4 => any::TypeId::of::<i64>(),
            5 => panic!(), // string
            6 => panic!(), // bool
            7 => any::TypeId::of::<i16>(),
            8 => panic!(), // complex64
            9 => any::TypeId::of::<i8>(),
            10 => panic!(), // float16
            11 => any::TypeId::of::<f64>(),
            12 => panic!(), // complex128
            _ => panic!(),
        }
    }
}

impl<'a> TensorRef<'a> {
    pub fn data<T: 'static>(&self) -> &[T] {
        assert_eq!(any::TypeId::of::<T>(), self.type_id);
        unsafe { slice::from_raw_parts(self.data as *const T, self.n_bytes / mem::size_of::<T>()) }
    }
}

impl<'a> TensorRefMut<'a> {
    pub fn data_mut<T: 'static>(&self) -> &mut [T] {
        assert_eq!(any::TypeId::of::<T>(), self.type_id);
        unsafe { slice::from_raw_parts_mut(self.data as *mut T, self.n_bytes / mem::size_of::<T>()) }
    }
}
