# A shared library for torch, produced on Ubuntu 16.04 with GCC 5.4 via a build from source
Following all instructions in the link below fresh anaconda2 environment.
I needed to amend the final step to:
`USE_MKL_DNN=0 USE_CUDA=0 USE_DISTRIBUTED=0 python setup.py install`
And don't forget to `python setup.py clean --all` if you rerun the above with different environment variables.
https://github.com/pytorch/pytorch#from-source

# Why?
I ran into problems linking together a drake prebuilt binary, which uses GCC >5 with new ABI, with 
the version of torchlib from the website, which current supports the old ABI.  Relevant links below:
https://stackoverflow.com/questions/53788767/why-do-i-get-linker-errors-when-i-build-a-cmake-project-using-drake-but-i-can-c/53788787#53788787
https://github.com/pytorch/pytorch/issues/14694
https://github.com/pytorch/pytorch/issues/14620
commands to check ABI version are:
`<nm -C/objdummp -TC> </path/to/.so> | grep std::string | wc -l` for old ABI
`<nm -C/ objdump -TC> </path/to/.so> | grep std::__cxx11 | wc -l` for new ABI
or 
```
python
import torch
torch.compiled_with_cxx11_abi()
```
