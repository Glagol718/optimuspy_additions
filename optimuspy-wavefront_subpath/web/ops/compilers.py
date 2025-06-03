from abc import ABCMeta, abstractmethod
from enum import Enum


class Singleton(ABCMeta):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        # else:
            # cls._instances[cls].__init__(*args, **kwargs)
        return cls._instances[cls]


class GenericCflags(Enum):
    def __str__(self) -> str:
        return self.name


class SubmitFormCflags(Enum):
    """Maximum number of cflags options used in submit form"""
    O0 = 0
    O1 = 1
    O2 = 2
    O3 = 3


class GCCCflags(GenericCflags):
    O0 = '-fopenmp -O0 -march=native'
    O1 = '-fopenmp -O1 -march=native'
    O2 = '-fopenmp -O2 -march=native'
    O3 = '-fopenmp -O3 -march=native'

# TODO: refactor!
class ClangCflags(GenericCflags):
    O0 = '-fopenmp -L/usr/lib/llvm-14/lib -O0 -march=native'
    O1 = '-fopenmp -L/usr/lib/llvm-14/lib -O1 -march=native'
    O2 = '-fopenmp -L/usr/lib/llvm-14/lib -O2 -march=native'
    O3 = '-fopenmp -L/usr/lib/llvm-14/lib -O3 -march=native'


class MSVCCflags(GenericCflags):
    O0 = '/O0 /arch:AVX2'
    O1 = '/O1 /arch:AVX2'
    O2 = '/O2 /arch:AVX2'


class Compiler(metaclass=Singleton):
    cflags = GCCCflags
    out: str = '-o'

    @property
    @abstractmethod
    def name(self):
        ...

    @property
    @abstractmethod
    def short(self):
        ...


class GCC(Compiler):
    @property
    def name(self):
        return 'g++'

    @property
    def short(self):
        return 'g++'


class Clang(Compiler):
    cflags = ClangCflags
    @property
    def name(self):
        return 'clang++-15'

    @property
    def short(self):
        return 'clang'


class MSVC(Compiler):
    cflags = MSVCCflags
    out: str = '/Fe:'

    @property
    def name(self):
        return 'cl'

    @property
    def short(self):
        return 'msvc'


class Compilers(Enum):
    GCC = 0
    Clang = 1
    MSVC = 2

    @property
    def obj(self) -> Compiler | None:
        match self:
            case Compilers.GCC:
                return GCC()
            case Compilers.Clang:
                return Clang()
            case Compilers.MSVC:
                return MSVC()
        return None
