# ── Optimatrix CUDA — Makefile ───────────────────────────────────────
#
# Usage :
#   make all            Compile tout
#   make test1          Phase 1 : Activations + Hadamard
#   make test2          Phase 2 : GEMV + GEMM
#   make test3          Phase 3 : Conv1D + Scan 1D
#   make test_scan2d    Phase 4 : Scan 2D (4 strategies)
#   make tests          Toutes les phases
#   make clean          Nettoyer
#
# Ajuster ARCH selon ta carte GPU :
#   sm_75  RTX 20xx / MX450 / T4
#   sm_86  RTX 30xx
#   sm_89  RTX 40xx
#   sm_90  H100

NVCC      = nvcc
ARCH      = -arch=sm_75
CFLAGS    = -O2 $(ARCH) -Iinclude
OBJDIR    = obj

# ── Objets ───────────────────────────────────────────────────────────

OBJ_ACT    = $(OBJDIR)/activations.o
OBJ_HAD    = $(OBJDIR)/hadamard.o
OBJ_GEMM   = $(OBJDIR)/gemm.o
OBJ_CONV   = $(OBJDIR)/conv1d.o
OBJ_S1D    = $(OBJDIR)/scan1d.o
OBJ_MAMBA  = $(OBJDIR)/mamba_block.o
OBJ_NAIVE  = $(OBJDIR)/scan2d_naive.o
OBJ_NVEC   = $(OBJDIR)/scan2d_naive_vec.o
OBJ_COOP   = $(OBJDIR)/scan2d_coop.o
OBJ_TILED  = $(OBJDIR)/scan2d_tiled.o

OBJS_ALL   = $(OBJ_ACT) $(OBJ_HAD) $(OBJ_GEMM) \
             $(OBJ_CONV) $(OBJ_S1D) $(OBJ_MAMBA) \
             $(OBJ_NAIVE) $(OBJ_NVEC) $(OBJ_COOP) $(OBJ_TILED)

# ── Regles de compilation ───────────────────────────────────────────

all: $(OBJS_ALL)

$(OBJDIR):
	mkdir -p $(OBJDIR)

$(OBJ_ACT): src/activations.cu include/optimatrix.h | $(OBJDIR)
	$(NVCC) $(CFLAGS) -c $< -o $@

$(OBJ_HAD): src/hadamard.cu include/optimatrix.h | $(OBJDIR)
	$(NVCC) $(CFLAGS) -c $< -o $@

$(OBJ_GEMM): src/gemm.cu include/optimatrix.h | $(OBJDIR)
	$(NVCC) $(CFLAGS) -c $< -o $@

$(OBJ_CONV): src/conv1d.cu include/optimatrix.h | $(OBJDIR)
	$(NVCC) $(CFLAGS) -c $< -o $@

$(OBJ_S1D): src/scan1d.cu include/optimatrix.h | $(OBJDIR)
	$(NVCC) $(CFLAGS) -c $< -o $@

$(OBJ_MAMBA): src/mamba_block.cu include/optimatrix.h | $(OBJDIR)
	$(NVCC) $(CFLAGS) -c $< -o $@

$(OBJ_NAIVE): src/scan2d/naive/scan2d_naive.cu include/optimatrix.h | $(OBJDIR)
	$(NVCC) $(CFLAGS) -c $< -o $@

$(OBJ_NVEC): src/scan2d/naive_vec/scan2d_naive_vec.cu include/optimatrix.h | $(OBJDIR)
	$(NVCC) $(CFLAGS) -c $< -o $@

$(OBJ_COOP): src/scan2d/coop/scan2d_coop.cu include/optimatrix.h | $(OBJDIR)
	$(NVCC) $(CFLAGS) -rdc=true -c $< -o $@

$(OBJ_TILED): src/scan2d/tiled/scan2d_tiled.cu include/optimatrix.h | $(OBJDIR)
	$(NVCC) $(CFLAGS) -c $< -o $@

# ── Phase 1 : Activations + Hadamard ────────────────────────────────

test1: $(OBJDIR)/test_activations $(OBJDIR)/test_hadamard
	@echo ""
	@echo "─── Phase 1 : Activations + Hadamard ───"
	./$(OBJDIR)/test_activations
	@echo ""
	./$(OBJDIR)/test_hadamard

$(OBJDIR)/test_activations: tests/test_activations.cu $(OBJ_ACT) | $(OBJDIR)
	$(NVCC) $(CFLAGS) $< $(OBJ_ACT) -o $@

$(OBJDIR)/test_hadamard: tests/test_hadamard.cu $(OBJ_HAD) | $(OBJDIR)
	$(NVCC) $(CFLAGS) $< $(OBJ_HAD) -o $@

# ── Phase 2 : GEMV + GEMM ───────────────────────────────────────────

test2: $(OBJDIR)/test_gemm
	@echo ""
	@echo "─── Phase 2 : GEMV + GEMM ───"
	./$(OBJDIR)/test_gemm

$(OBJDIR)/test_gemm: tests/test_gemm.cu $(OBJ_GEMM) | $(OBJDIR)
	$(NVCC) $(CFLAGS) $< $(OBJ_GEMM) -o $@

# ── Phase 3 : Conv1D + Scan 1D ──────────────────────────────────────

test3: $(OBJDIR)/test_conv1d $(OBJDIR)/test_scan1d
	@echo ""
	@echo "─── Phase 3 : Conv1D ───"
	./$(OBJDIR)/test_conv1d
	@echo ""
	@echo "─── Phase 3 : Scan 1D (seq + Blelloch) ───"
	./$(OBJDIR)/test_scan1d

$(OBJDIR)/test_conv1d: tests/test_conv1d.cu $(OBJ_CONV) | $(OBJDIR)
	$(NVCC) $(CFLAGS) $< $(OBJ_CONV) -o $@

$(OBJDIR)/test_scan1d: tests/test_scan1d.cu $(OBJ_S1D) | $(OBJDIR)
	$(NVCC) $(CFLAGS) $< $(OBJ_S1D) -o $@

# ── Phase 4 : Scan 2D — 4 strategies ────────────────────────────────

SCAN2D_OBJS = $(OBJ_NAIVE) $(OBJ_NVEC) $(OBJ_COOP) $(OBJ_TILED)

test_scan2d: $(OBJDIR)/test_scan2d
	@echo ""
	@echo "─── Phase 4 : Scan 2D — Correctness + Benchmark ───"
	./$(OBJDIR)/test_scan2d

$(OBJDIR)/test_scan2d: tests/test_scan2d.cu $(SCAN2D_OBJS) | $(OBJDIR)
	$(NVCC) $(CFLAGS) -rdc=true $< $(SCAN2D_OBJS) -lcudadevrt -o $@

# ── Toutes les phases ───────────────────────────────────────────────

MAMBA_OBJS = $(OBJ_ACT) $(OBJ_HAD) $(OBJ_GEMM) \
             $(OBJ_CONV) $(OBJ_S1D) $(OBJ_MAMBA)

test5: $(OBJDIR)/test_mamba_block
	@echo ""
	@echo "─── Phase 5 : MambaBlock forward complet ───"
	./$(OBJDIR)/test_mamba_block

$(OBJDIR)/test_mamba_block: tests/test_mamba_block.cu $(MAMBA_OBJS) | $(OBJDIR)
	$(NVCC) $(CFLAGS) $< $(MAMBA_OBJS) -o $@

tests: test1 test2 test3 test_scan2d test5

# ── Clean ────────────────────────────────────────────────────────────

clean:
	rm -rf $(OBJDIR)

.PHONY: all clean test1 test2 test3 test_scan2d tests
