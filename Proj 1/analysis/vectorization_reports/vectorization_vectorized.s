	.text
	.def	@feat.00;
	.scl	3;
	.type	0;
	.endef
	.globl	@feat.00
.set @feat.00, 1
	.file	"test_kernels.c"
	.def	_axpy_scalar_f32;
	.scl	2;
	.type	32;
	.endef
	.globl	_axpy_scalar_f32                # -- Begin function axpy_scalar_f32
	.p2align	4, 0x90
_axpy_scalar_f32:                       # @axpy_scalar_f32
# %bb.0:
	pushl	%ebx
	pushl	%edi
	pushl	%esi
	movl	28(%esp), %eax
	testl	%eax, %eax
	je	LBB0_11
# %bb.1:
	movl	24(%esp), %ecx
	movl	20(%esp), %edx
	vmovss	16(%esp), %xmm0                 # xmm0 = mem[0],zero,zero,zero
	xorl	%esi, %esi
	cmpl	$31, %eax
	jbe	LBB0_2
# %bb.7:
	leal	(%ecx,%eax,4), %edi
	leal	(%edx,%eax,4), %ebx
	cmpl	%ecx, %ebx
	seta	%bl
	cmpl	%edx, %edi
	seta	%bh
	testb	%bh, %bl
	jne	LBB0_2
# %bb.8:
	movl	%eax, %esi
	andl	$-32, %esi
	vbroadcastss	%xmm0, %ymm1
	xorl	%edi, %edi
	.p2align	4, 0x90
LBB0_9:                                 # =>This Inner Loop Header: Depth=1
	vmovups	(%edx,%edi,4), %ymm2
	vmovups	32(%edx,%edi,4), %ymm3
	vmovups	64(%edx,%edi,4), %ymm4
	vmovups	96(%edx,%edi,4), %ymm5
	vfmadd213ps	(%ecx,%edi,4), %ymm1, %ymm2 # ymm2 = (ymm1 * ymm2) + mem
	vfmadd213ps	32(%ecx,%edi,4), %ymm1, %ymm3 # ymm3 = (ymm1 * ymm3) + mem
	vfmadd213ps	64(%ecx,%edi,4), %ymm1, %ymm4 # ymm4 = (ymm1 * ymm4) + mem
	vfmadd213ps	96(%ecx,%edi,4), %ymm1, %ymm5 # ymm5 = (ymm1 * ymm5) + mem
	vmovups	%ymm2, (%ecx,%edi,4)
	vmovups	%ymm3, 32(%ecx,%edi,4)
	vmovups	%ymm4, 64(%ecx,%edi,4)
	vmovups	%ymm5, 96(%ecx,%edi,4)
	addl	$32, %edi
	cmpl	%edi, %esi
	jne	LBB0_9
# %bb.10:
	cmpl	%eax, %esi
	je	LBB0_11
LBB0_2:
	movl	%eax, %ebx
	movl	%esi, %edi
	andl	$7, %ebx
	je	LBB0_5
# %bb.3:
	movl	%esi, %edi
	.p2align	4, 0x90
LBB0_4:                                 # =>This Inner Loop Header: Depth=1
	vmovss	(%edx,%edi,4), %xmm1            # xmm1 = mem[0],zero,zero,zero
	vfmadd213ss	(%ecx,%edi,4), %xmm0, %xmm1 # xmm1 = (xmm0 * xmm1) + mem
	vmovss	%xmm1, (%ecx,%edi,4)
	incl	%edi
	decl	%ebx
	jne	LBB0_4
LBB0_5:
	subl	%eax, %esi
	cmpl	$-8, %esi
	ja	LBB0_11
	.p2align	4, 0x90
LBB0_6:                                 # =>This Inner Loop Header: Depth=1
	vmovss	(%edx,%edi,4), %xmm1            # xmm1 = mem[0],zero,zero,zero
	vfmadd213ss	(%ecx,%edi,4), %xmm0, %xmm1 # xmm1 = (xmm0 * xmm1) + mem
	vmovss	%xmm1, (%ecx,%edi,4)
	vmovss	4(%edx,%edi,4), %xmm1           # xmm1 = mem[0],zero,zero,zero
	vfmadd213ss	4(%ecx,%edi,4), %xmm0, %xmm1 # xmm1 = (xmm0 * xmm1) + mem
	vmovss	%xmm1, 4(%ecx,%edi,4)
	vmovss	8(%edx,%edi,4), %xmm1           # xmm1 = mem[0],zero,zero,zero
	vfmadd213ss	8(%ecx,%edi,4), %xmm0, %xmm1 # xmm1 = (xmm0 * xmm1) + mem
	vmovss	%xmm1, 8(%ecx,%edi,4)
	vmovss	12(%edx,%edi,4), %xmm1          # xmm1 = mem[0],zero,zero,zero
	vfmadd213ss	12(%ecx,%edi,4), %xmm0, %xmm1 # xmm1 = (xmm0 * xmm1) + mem
	vmovss	%xmm1, 12(%ecx,%edi,4)
	vmovss	16(%edx,%edi,4), %xmm1          # xmm1 = mem[0],zero,zero,zero
	vfmadd213ss	16(%ecx,%edi,4), %xmm0, %xmm1 # xmm1 = (xmm0 * xmm1) + mem
	vmovss	%xmm1, 16(%ecx,%edi,4)
	vmovss	20(%edx,%edi,4), %xmm1          # xmm1 = mem[0],zero,zero,zero
	vfmadd213ss	20(%ecx,%edi,4), %xmm0, %xmm1 # xmm1 = (xmm0 * xmm1) + mem
	vmovss	%xmm1, 20(%ecx,%edi,4)
	vmovss	24(%edx,%edi,4), %xmm1          # xmm1 = mem[0],zero,zero,zero
	vfmadd213ss	24(%ecx,%edi,4), %xmm0, %xmm1 # xmm1 = (xmm0 * xmm1) + mem
	vmovss	%xmm1, 24(%ecx,%edi,4)
	vmovss	28(%edx,%edi,4), %xmm1          # xmm1 = mem[0],zero,zero,zero
	vfmadd213ss	28(%ecx,%edi,4), %xmm0, %xmm1 # xmm1 = (xmm0 * xmm1) + mem
	vmovss	%xmm1, 28(%ecx,%edi,4)
	addl	$8, %edi
	cmpl	%edi, %eax
	jne	LBB0_6
LBB0_11:
	popl	%esi
	popl	%edi
	popl	%ebx
	vzeroupper
	retl
                                        # -- End function
	.def	_axpy_vectorized_f32;
	.scl	2;
	.type	32;
	.endef
	.globl	_axpy_vectorized_f32            # -- Begin function axpy_vectorized_f32
	.p2align	4, 0x90
_axpy_vectorized_f32:                   # @axpy_vectorized_f32
# %bb.0:
	pushl	%ebx
	pushl	%edi
	pushl	%esi
	movl	28(%esp), %eax
	testl	%eax, %eax
	je	LBB1_11
# %bb.1:
	movl	24(%esp), %ecx
	movl	20(%esp), %edx
	vmovss	16(%esp), %xmm0                 # xmm0 = mem[0],zero,zero,zero
	xorl	%esi, %esi
	cmpl	$31, %eax
	jbe	LBB1_2
# %bb.7:
	leal	(%ecx,%eax,4), %edi
	leal	(%edx,%eax,4), %ebx
	cmpl	%ecx, %ebx
	seta	%bl
	cmpl	%edx, %edi
	seta	%bh
	testb	%bh, %bl
	jne	LBB1_2
# %bb.8:
	movl	%eax, %esi
	andl	$-32, %esi
	vbroadcastss	%xmm0, %ymm1
	xorl	%edi, %edi
	.p2align	4, 0x90
LBB1_9:                                 # =>This Inner Loop Header: Depth=1
	vmovups	(%edx,%edi,4), %ymm2
	vmovups	32(%edx,%edi,4), %ymm3
	vmovups	64(%edx,%edi,4), %ymm4
	vmovups	96(%edx,%edi,4), %ymm5
	vfmadd213ps	(%ecx,%edi,4), %ymm1, %ymm2 # ymm2 = (ymm1 * ymm2) + mem
	vfmadd213ps	32(%ecx,%edi,4), %ymm1, %ymm3 # ymm3 = (ymm1 * ymm3) + mem
	vfmadd213ps	64(%ecx,%edi,4), %ymm1, %ymm4 # ymm4 = (ymm1 * ymm4) + mem
	vfmadd213ps	96(%ecx,%edi,4), %ymm1, %ymm5 # ymm5 = (ymm1 * ymm5) + mem
	vmovups	%ymm2, (%ecx,%edi,4)
	vmovups	%ymm3, 32(%ecx,%edi,4)
	vmovups	%ymm4, 64(%ecx,%edi,4)
	vmovups	%ymm5, 96(%ecx,%edi,4)
	addl	$32, %edi
	cmpl	%edi, %esi
	jne	LBB1_9
# %bb.10:
	cmpl	%eax, %esi
	je	LBB1_11
LBB1_2:
	movl	%eax, %ebx
	movl	%esi, %edi
	andl	$7, %ebx
	je	LBB1_5
# %bb.3:
	movl	%esi, %edi
	.p2align	4, 0x90
LBB1_4:                                 # =>This Inner Loop Header: Depth=1
	vmovss	(%edx,%edi,4), %xmm1            # xmm1 = mem[0],zero,zero,zero
	vfmadd213ss	(%ecx,%edi,4), %xmm0, %xmm1 # xmm1 = (xmm0 * xmm1) + mem
	vmovss	%xmm1, (%ecx,%edi,4)
	incl	%edi
	decl	%ebx
	jne	LBB1_4
LBB1_5:
	subl	%eax, %esi
	cmpl	$-8, %esi
	ja	LBB1_11
	.p2align	4, 0x90
LBB1_6:                                 # =>This Inner Loop Header: Depth=1
	vmovss	(%edx,%edi,4), %xmm1            # xmm1 = mem[0],zero,zero,zero
	vfmadd213ss	(%ecx,%edi,4), %xmm0, %xmm1 # xmm1 = (xmm0 * xmm1) + mem
	vmovss	%xmm1, (%ecx,%edi,4)
	vmovss	4(%edx,%edi,4), %xmm1           # xmm1 = mem[0],zero,zero,zero
	vfmadd213ss	4(%ecx,%edi,4), %xmm0, %xmm1 # xmm1 = (xmm0 * xmm1) + mem
	vmovss	%xmm1, 4(%ecx,%edi,4)
	vmovss	8(%edx,%edi,4), %xmm1           # xmm1 = mem[0],zero,zero,zero
	vfmadd213ss	8(%ecx,%edi,4), %xmm0, %xmm1 # xmm1 = (xmm0 * xmm1) + mem
	vmovss	%xmm1, 8(%ecx,%edi,4)
	vmovss	12(%edx,%edi,4), %xmm1          # xmm1 = mem[0],zero,zero,zero
	vfmadd213ss	12(%ecx,%edi,4), %xmm0, %xmm1 # xmm1 = (xmm0 * xmm1) + mem
	vmovss	%xmm1, 12(%ecx,%edi,4)
	vmovss	16(%edx,%edi,4), %xmm1          # xmm1 = mem[0],zero,zero,zero
	vfmadd213ss	16(%ecx,%edi,4), %xmm0, %xmm1 # xmm1 = (xmm0 * xmm1) + mem
	vmovss	%xmm1, 16(%ecx,%edi,4)
	vmovss	20(%edx,%edi,4), %xmm1          # xmm1 = mem[0],zero,zero,zero
	vfmadd213ss	20(%ecx,%edi,4), %xmm0, %xmm1 # xmm1 = (xmm0 * xmm1) + mem
	vmovss	%xmm1, 20(%ecx,%edi,4)
	vmovss	24(%edx,%edi,4), %xmm1          # xmm1 = mem[0],zero,zero,zero
	vfmadd213ss	24(%ecx,%edi,4), %xmm0, %xmm1 # xmm1 = (xmm0 * xmm1) + mem
	vmovss	%xmm1, 24(%ecx,%edi,4)
	vmovss	28(%edx,%edi,4), %xmm1          # xmm1 = mem[0],zero,zero,zero
	vfmadd213ss	28(%ecx,%edi,4), %xmm0, %xmm1 # xmm1 = (xmm0 * xmm1) + mem
	vmovss	%xmm1, 28(%ecx,%edi,4)
	addl	$8, %edi
	cmpl	%edi, %eax
	jne	LBB1_6
LBB1_11:
	popl	%esi
	popl	%edi
	popl	%ebx
	vzeroupper
	retl
                                        # -- End function
	.def	_axpy_scalar_f64;
	.scl	2;
	.type	32;
	.endef
	.globl	_axpy_scalar_f64                # -- Begin function axpy_scalar_f64
	.p2align	4, 0x90
_axpy_scalar_f64:                       # @axpy_scalar_f64
# %bb.0:
	pushl	%ebx
	pushl	%edi
	pushl	%esi
	movl	32(%esp), %eax
	testl	%eax, %eax
	je	LBB2_11
# %bb.1:
	movl	28(%esp), %ecx
	movl	24(%esp), %edx
	vmovsd	16(%esp), %xmm0                 # xmm0 = mem[0],zero
	xorl	%esi, %esi
	cmpl	$15, %eax
	jbe	LBB2_2
# %bb.7:
	leal	(%ecx,%eax,8), %edi
	leal	(%edx,%eax,8), %ebx
	cmpl	%ecx, %ebx
	seta	%bl
	cmpl	%edx, %edi
	seta	%bh
	testb	%bh, %bl
	jne	LBB2_2
# %bb.8:
	movl	%eax, %esi
	andl	$-16, %esi
	vbroadcastsd	%xmm0, %ymm1
	xorl	%edi, %edi
	.p2align	4, 0x90
LBB2_9:                                 # =>This Inner Loop Header: Depth=1
	vmovupd	(%edx,%edi,8), %ymm2
	vmovupd	32(%edx,%edi,8), %ymm3
	vmovupd	64(%edx,%edi,8), %ymm4
	vmovupd	96(%edx,%edi,8), %ymm5
	vfmadd213pd	(%ecx,%edi,8), %ymm1, %ymm2 # ymm2 = (ymm1 * ymm2) + mem
	vfmadd213pd	32(%ecx,%edi,8), %ymm1, %ymm3 # ymm3 = (ymm1 * ymm3) + mem
	vfmadd213pd	64(%ecx,%edi,8), %ymm1, %ymm4 # ymm4 = (ymm1 * ymm4) + mem
	vfmadd213pd	96(%ecx,%edi,8), %ymm1, %ymm5 # ymm5 = (ymm1 * ymm5) + mem
	vmovupd	%ymm2, (%ecx,%edi,8)
	vmovupd	%ymm3, 32(%ecx,%edi,8)
	vmovupd	%ymm4, 64(%ecx,%edi,8)
	vmovupd	%ymm5, 96(%ecx,%edi,8)
	addl	$16, %edi
	cmpl	%edi, %esi
	jne	LBB2_9
# %bb.10:
	cmpl	%eax, %esi
	je	LBB2_11
LBB2_2:
	movl	%eax, %ebx
	movl	%esi, %edi
	andl	$7, %ebx
	je	LBB2_5
# %bb.3:
	movl	%esi, %edi
	.p2align	4, 0x90
LBB2_4:                                 # =>This Inner Loop Header: Depth=1
	vmovsd	(%edx,%edi,8), %xmm1            # xmm1 = mem[0],zero
	vfmadd213sd	(%ecx,%edi,8), %xmm0, %xmm1 # xmm1 = (xmm0 * xmm1) + mem
	vmovsd	%xmm1, (%ecx,%edi,8)
	incl	%edi
	decl	%ebx
	jne	LBB2_4
LBB2_5:
	subl	%eax, %esi
	cmpl	$-8, %esi
	ja	LBB2_11
	.p2align	4, 0x90
LBB2_6:                                 # =>This Inner Loop Header: Depth=1
	vmovsd	(%edx,%edi,8), %xmm1            # xmm1 = mem[0],zero
	vfmadd213sd	(%ecx,%edi,8), %xmm0, %xmm1 # xmm1 = (xmm0 * xmm1) + mem
	vmovsd	%xmm1, (%ecx,%edi,8)
	vmovsd	8(%edx,%edi,8), %xmm1           # xmm1 = mem[0],zero
	vfmadd213sd	8(%ecx,%edi,8), %xmm0, %xmm1 # xmm1 = (xmm0 * xmm1) + mem
	vmovsd	%xmm1, 8(%ecx,%edi,8)
	vmovsd	16(%edx,%edi,8), %xmm1          # xmm1 = mem[0],zero
	vfmadd213sd	16(%ecx,%edi,8), %xmm0, %xmm1 # xmm1 = (xmm0 * xmm1) + mem
	vmovsd	%xmm1, 16(%ecx,%edi,8)
	vmovsd	24(%edx,%edi,8), %xmm1          # xmm1 = mem[0],zero
	vfmadd213sd	24(%ecx,%edi,8), %xmm0, %xmm1 # xmm1 = (xmm0 * xmm1) + mem
	vmovsd	%xmm1, 24(%ecx,%edi,8)
	vmovsd	32(%edx,%edi,8), %xmm1          # xmm1 = mem[0],zero
	vfmadd213sd	32(%ecx,%edi,8), %xmm0, %xmm1 # xmm1 = (xmm0 * xmm1) + mem
	vmovsd	%xmm1, 32(%ecx,%edi,8)
	vmovsd	40(%edx,%edi,8), %xmm1          # xmm1 = mem[0],zero
	vfmadd213sd	40(%ecx,%edi,8), %xmm0, %xmm1 # xmm1 = (xmm0 * xmm1) + mem
	vmovsd	%xmm1, 40(%ecx,%edi,8)
	vmovsd	48(%edx,%edi,8), %xmm1          # xmm1 = mem[0],zero
	vfmadd213sd	48(%ecx,%edi,8), %xmm0, %xmm1 # xmm1 = (xmm0 * xmm1) + mem
	vmovsd	%xmm1, 48(%ecx,%edi,8)
	vmovsd	56(%edx,%edi,8), %xmm1          # xmm1 = mem[0],zero
	vfmadd213sd	56(%ecx,%edi,8), %xmm0, %xmm1 # xmm1 = (xmm0 * xmm1) + mem
	vmovsd	%xmm1, 56(%ecx,%edi,8)
	addl	$8, %edi
	cmpl	%edi, %eax
	jne	LBB2_6
LBB2_11:
	popl	%esi
	popl	%edi
	popl	%ebx
	vzeroupper
	retl
                                        # -- End function
	.def	_axpy_vectorized_f64;
	.scl	2;
	.type	32;
	.endef
	.globl	_axpy_vectorized_f64            # -- Begin function axpy_vectorized_f64
	.p2align	4, 0x90
_axpy_vectorized_f64:                   # @axpy_vectorized_f64
# %bb.0:
	pushl	%ebx
	pushl	%edi
	pushl	%esi
	movl	32(%esp), %eax
	testl	%eax, %eax
	je	LBB3_11
# %bb.1:
	movl	28(%esp), %ecx
	movl	24(%esp), %edx
	vmovsd	16(%esp), %xmm0                 # xmm0 = mem[0],zero
	xorl	%esi, %esi
	cmpl	$15, %eax
	jbe	LBB3_2
# %bb.7:
	leal	(%ecx,%eax,8), %edi
	leal	(%edx,%eax,8), %ebx
	cmpl	%ecx, %ebx
	seta	%bl
	cmpl	%edx, %edi
	seta	%bh
	testb	%bh, %bl
	jne	LBB3_2
# %bb.8:
	movl	%eax, %esi
	andl	$-16, %esi
	vbroadcastsd	%xmm0, %ymm1
	xorl	%edi, %edi
	.p2align	4, 0x90
LBB3_9:                                 # =>This Inner Loop Header: Depth=1
	vmovupd	(%edx,%edi,8), %ymm2
	vmovupd	32(%edx,%edi,8), %ymm3
	vmovupd	64(%edx,%edi,8), %ymm4
	vmovupd	96(%edx,%edi,8), %ymm5
	vfmadd213pd	(%ecx,%edi,8), %ymm1, %ymm2 # ymm2 = (ymm1 * ymm2) + mem
	vfmadd213pd	32(%ecx,%edi,8), %ymm1, %ymm3 # ymm3 = (ymm1 * ymm3) + mem
	vfmadd213pd	64(%ecx,%edi,8), %ymm1, %ymm4 # ymm4 = (ymm1 * ymm4) + mem
	vfmadd213pd	96(%ecx,%edi,8), %ymm1, %ymm5 # ymm5 = (ymm1 * ymm5) + mem
	vmovupd	%ymm2, (%ecx,%edi,8)
	vmovupd	%ymm3, 32(%ecx,%edi,8)
	vmovupd	%ymm4, 64(%ecx,%edi,8)
	vmovupd	%ymm5, 96(%ecx,%edi,8)
	addl	$16, %edi
	cmpl	%edi, %esi
	jne	LBB3_9
# %bb.10:
	cmpl	%eax, %esi
	je	LBB3_11
LBB3_2:
	movl	%eax, %ebx
	movl	%esi, %edi
	andl	$7, %ebx
	je	LBB3_5
# %bb.3:
	movl	%esi, %edi
	.p2align	4, 0x90
LBB3_4:                                 # =>This Inner Loop Header: Depth=1
	vmovsd	(%edx,%edi,8), %xmm1            # xmm1 = mem[0],zero
	vfmadd213sd	(%ecx,%edi,8), %xmm0, %xmm1 # xmm1 = (xmm0 * xmm1) + mem
	vmovsd	%xmm1, (%ecx,%edi,8)
	incl	%edi
	decl	%ebx
	jne	LBB3_4
LBB3_5:
	subl	%eax, %esi
	cmpl	$-8, %esi
	ja	LBB3_11
	.p2align	4, 0x90
LBB3_6:                                 # =>This Inner Loop Header: Depth=1
	vmovsd	(%edx,%edi,8), %xmm1            # xmm1 = mem[0],zero
	vfmadd213sd	(%ecx,%edi,8), %xmm0, %xmm1 # xmm1 = (xmm0 * xmm1) + mem
	vmovsd	%xmm1, (%ecx,%edi,8)
	vmovsd	8(%edx,%edi,8), %xmm1           # xmm1 = mem[0],zero
	vfmadd213sd	8(%ecx,%edi,8), %xmm0, %xmm1 # xmm1 = (xmm0 * xmm1) + mem
	vmovsd	%xmm1, 8(%ecx,%edi,8)
	vmovsd	16(%edx,%edi,8), %xmm1          # xmm1 = mem[0],zero
	vfmadd213sd	16(%ecx,%edi,8), %xmm0, %xmm1 # xmm1 = (xmm0 * xmm1) + mem
	vmovsd	%xmm1, 16(%ecx,%edi,8)
	vmovsd	24(%edx,%edi,8), %xmm1          # xmm1 = mem[0],zero
	vfmadd213sd	24(%ecx,%edi,8), %xmm0, %xmm1 # xmm1 = (xmm0 * xmm1) + mem
	vmovsd	%xmm1, 24(%ecx,%edi,8)
	vmovsd	32(%edx,%edi,8), %xmm1          # xmm1 = mem[0],zero
	vfmadd213sd	32(%ecx,%edi,8), %xmm0, %xmm1 # xmm1 = (xmm0 * xmm1) + mem
	vmovsd	%xmm1, 32(%ecx,%edi,8)
	vmovsd	40(%edx,%edi,8), %xmm1          # xmm1 = mem[0],zero
	vfmadd213sd	40(%ecx,%edi,8), %xmm0, %xmm1 # xmm1 = (xmm0 * xmm1) + mem
	vmovsd	%xmm1, 40(%ecx,%edi,8)
	vmovsd	48(%edx,%edi,8), %xmm1          # xmm1 = mem[0],zero
	vfmadd213sd	48(%ecx,%edi,8), %xmm0, %xmm1 # xmm1 = (xmm0 * xmm1) + mem
	vmovsd	%xmm1, 48(%ecx,%edi,8)
	vmovsd	56(%edx,%edi,8), %xmm1          # xmm1 = mem[0],zero
	vfmadd213sd	56(%ecx,%edi,8), %xmm0, %xmm1 # xmm1 = (xmm0 * xmm1) + mem
	vmovsd	%xmm1, 56(%ecx,%edi,8)
	addl	$8, %edi
	cmpl	%edi, %eax
	jne	LBB3_6
LBB3_11:
	popl	%esi
	popl	%edi
	popl	%ebx
	vzeroupper
	retl
                                        # -- End function
	.def	_dot_product_scalar_f32;
	.scl	2;
	.type	32;
	.endef
	.globl	_dot_product_scalar_f32         # -- Begin function dot_product_scalar_f32
	.p2align	4, 0x90
_dot_product_scalar_f32:                # @dot_product_scalar_f32
# %bb.0:
	pushl	%ebx
	pushl	%edi
	pushl	%esi
	movl	28(%esp), %eax
	movl	24(%esp), %ecx
	testl	%ecx, %ecx
	je	LBB4_1
# %bb.2:
	movl	20(%esp), %edx
	movl	16(%esp), %esi
	cmpl	$15, %ecx
	ja	LBB4_4
# %bb.3:
	vxorps	%xmm0, %xmm0, %xmm0
	xorl	%edi, %edi
	jmp	LBB4_7
LBB4_1:
	vxorps	%xmm0, %xmm0, %xmm0
	jmp	LBB4_8
LBB4_4:
	movl	%ecx, %edi
	andl	$-16, %edi
	vxorps	%xmm0, %xmm0, %xmm0
	xorl	%ebx, %ebx
	vxorps	%xmm1, %xmm1, %xmm1
	.p2align	4, 0x90
LBB4_5:                                 # =>This Inner Loop Header: Depth=1
	vmovups	(%edx,%ebx,4), %ymm2
	vmovups	32(%edx,%ebx,4), %ymm3
	vfmadd231ps	(%esi,%ebx,4), %ymm2, %ymm0 # ymm0 = (ymm2 * mem) + ymm0
	vfmadd231ps	32(%esi,%ebx,4), %ymm3, %ymm1 # ymm1 = (ymm3 * mem) + ymm1
	addl	$16, %ebx
	cmpl	%ebx, %edi
	jne	LBB4_5
# %bb.6:
	vaddps	%ymm0, %ymm1, %ymm0
	vextractf128	$1, %ymm0, %xmm1
	vaddps	%xmm1, %xmm0, %xmm0
	vshufpd	$1, %xmm0, %xmm0, %xmm1         # xmm1 = xmm0[1,0]
	vaddps	%xmm1, %xmm0, %xmm0
	vmovshdup	%xmm0, %xmm1            # xmm1 = xmm0[1,1,3,3]
	vaddss	%xmm1, %xmm0, %xmm0
	cmpl	%ecx, %edi
	je	LBB4_8
	.p2align	4, 0x90
LBB4_7:                                 # =>This Inner Loop Header: Depth=1
	vmovss	(%edx,%edi,4), %xmm1            # xmm1 = mem[0],zero,zero,zero
	vfmadd231ss	(%esi,%edi,4), %xmm1, %xmm0 # xmm0 = (xmm1 * mem) + xmm0
	incl	%edi
	cmpl	%edi, %ecx
	jne	LBB4_7
LBB4_8:
	vmovss	%xmm0, (%eax)
	popl	%esi
	popl	%edi
	popl	%ebx
	vzeroupper
	retl
                                        # -- End function
	.def	_dot_product_vectorized_f32;
	.scl	2;
	.type	32;
	.endef
	.globl	_dot_product_vectorized_f32     # -- Begin function dot_product_vectorized_f32
	.p2align	4, 0x90
_dot_product_vectorized_f32:            # @dot_product_vectorized_f32
# %bb.0:
	pushl	%ebx
	pushl	%edi
	pushl	%esi
	movl	28(%esp), %eax
	movl	24(%esp), %ecx
	testl	%ecx, %ecx
	je	LBB5_1
# %bb.2:
	movl	20(%esp), %edx
	movl	16(%esp), %esi
	cmpl	$15, %ecx
	ja	LBB5_4
# %bb.3:
	vxorps	%xmm0, %xmm0, %xmm0
	xorl	%edi, %edi
	jmp	LBB5_7
LBB5_1:
	vxorps	%xmm0, %xmm0, %xmm0
	jmp	LBB5_8
LBB5_4:
	movl	%ecx, %edi
	andl	$-16, %edi
	vxorps	%xmm0, %xmm0, %xmm0
	xorl	%ebx, %ebx
	vxorps	%xmm1, %xmm1, %xmm1
	.p2align	4, 0x90
LBB5_5:                                 # =>This Inner Loop Header: Depth=1
	vmovups	(%edx,%ebx,4), %ymm2
	vmovups	32(%edx,%ebx,4), %ymm3
	vfmadd231ps	(%esi,%ebx,4), %ymm2, %ymm0 # ymm0 = (ymm2 * mem) + ymm0
	vfmadd231ps	32(%esi,%ebx,4), %ymm3, %ymm1 # ymm1 = (ymm3 * mem) + ymm1
	addl	$16, %ebx
	cmpl	%ebx, %edi
	jne	LBB5_5
# %bb.6:
	vaddps	%ymm0, %ymm1, %ymm0
	vextractf128	$1, %ymm0, %xmm1
	vaddps	%xmm1, %xmm0, %xmm0
	vshufpd	$1, %xmm0, %xmm0, %xmm1         # xmm1 = xmm0[1,0]
	vaddps	%xmm1, %xmm0, %xmm0
	vmovshdup	%xmm0, %xmm1            # xmm1 = xmm0[1,1,3,3]
	vaddss	%xmm1, %xmm0, %xmm0
	cmpl	%ecx, %edi
	je	LBB5_8
	.p2align	4, 0x90
LBB5_7:                                 # =>This Inner Loop Header: Depth=1
	vmovss	(%edx,%edi,4), %xmm1            # xmm1 = mem[0],zero,zero,zero
	vfmadd231ss	(%esi,%edi,4), %xmm1, %xmm0 # xmm0 = (xmm1 * mem) + xmm0
	incl	%edi
	cmpl	%edi, %ecx
	jne	LBB5_7
LBB5_8:
	vmovss	%xmm0, (%eax)
	popl	%esi
	popl	%edi
	popl	%ebx
	vzeroupper
	retl
                                        # -- End function
	.def	_test_kernels;
	.scl	2;
	.type	32;
	.endef
	.globl	_test_kernels                   # -- Begin function test_kernels
	.p2align	4, 0x90
_test_kernels:                          # @test_kernels
# %bb.0:
	retl
                                        # -- End function
	.def	_main;
	.scl	2;
	.type	32;
	.endef
	.globl	_main                           # -- Begin function main
	.p2align	4, 0x90
_main:                                  # @main
# %bb.0:
	xorl	%eax, %eax
	retl
                                        # -- End function
	.addrsig
	.globl	__fltused
