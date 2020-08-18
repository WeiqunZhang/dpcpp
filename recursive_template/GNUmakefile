DPCPP_DIR ?= /opt/intel/inteloneapi/compiler/latest/linux

CEXE_sources = wp_parser_c.cpp wp_parser.lex.cpp wp_parser.tab.cpp wp_parser_y.cpp WarpXParser.cpp

USE_CUDA = FALSE
ifeq ($(USE_CUDA),TRUE)
    MYEXE = main-cuda.ex
    CEXE_sources += main-cuda.cpp
    objEXETempDir = tmp_build_dir/cuda
    LINKER = nvcc
    CXX = nvcc
    CXXFLAGS = -ccbin=g++ -Xcompiler=' -g -O3 -std=c++14' --std=c++14 -Wno-deprecated-gpu-targets -m64 -arch=compute_60 -code=sm_60 -maxrregcount=255 --expt-relaxed-constexpr --expt-extended-lambda -lineinfo --ptxas-options=-O3 --ptxas-options=-v --use_fast_math -dc -x cu
    CXXFLAGS += -DAMREX_USE_CUDA
    LINKFLAGS = -Wno-deprecated-gpu-targets -m64 -arch=compute_60 -code=sm_60 -maxrregcount=255 --expt-relaxed-constexpr --expt-extended-lambda -lineinfo --ptxas-options=-O3 --ptxas-options=-v --use_fast_math
    libraries =
else
    MYEXE = main-dpcpp.ex
    CEXE_sources += main-dpcpp.cpp
    objEXETempDir = tmp_build_dir/dpcpp
    LINKER = dpcpp
    CXX = dpcpp
    CXXFLAGS = -O3 -fsycl -fsycl-unnamed-lambda
    CXXFLAGS += -DAMREX_USE_DPCPP
    LINKFLAGS += -device-math-lib=fp32,fp64
    libraries =
endif

objStandalone = $(CEXE_sources:%.cpp=%.o)
objForExecs = $(addprefix $(objEXETempDir)/, $(objStandalone))
vpath %.cpp .
vpath %.H .
includes = -I.

default: $(MYEXE)

$(MYEXE): $(objForExecs)
	@echo Linking $@ ...
	$(LINKER) $(LINKFLAGS) $(includes) -o $@ $^ $(libraries)

$(objEXETempDir)/%.o: %.cpp
	@echo Compiling $*.cpp ...
	@if [ ! -d $(objEXETempDir) ]; then mkdir -p $(objEXETempDir); fi
	$(CXX) $(CXXFLAGS) $(includes) -c $< -o $@

$(objEXETempDir)/%.d: %.cpp
	@echo Depending $< ...
	@if [ ! -d $(objEXETempDir) ]; then mkdir -p $(objEXETempDir); fi
	@$(SHELL) -ec './mkdep $(includes) $< | \
		sed -e '\''s,^[^:]*\/,,'\'' | \
		sed -e '\''s,$*.o,$(objEXETempDir)/& $@,'\'' > $@'

clean::
	${RM} *.ex $(objEXETempDir)/*.o $(objEXETempDir)/*.d

realclean:: clean
	$(RM) -r tmp_build_dir

FORCE:

.PHONY: clean realclean

ifneq ($(MAKECMDGOALS),realclean)
ifneq ($(MAKECMDGOALS),clean)
ifneq ($(findstring print-,$(MAKECMDGOALS)),print-)
-include $(CEXE_sources:%.cpp=$(objEXETempDir)/%.d)
endif
endif
endif

print-%:
	@echo $* is '$($*)'
	@echo '    origin = $(origin $*)'
	@echo '     value = $(value  $*)'
