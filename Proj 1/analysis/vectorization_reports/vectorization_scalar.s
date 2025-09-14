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
	pushl	%esi
	movl	24(%esp), %ecx
	flds	12(%esp)
	testl	%ecx, %ecx
	je	LBB0_6
# %bb.1:
	movl	20(%esp), %eax
	movl	16(%esp), %edx
	xorl	%esi, %esi
	cmpl	$1, %ecx
	je	LBB0_4
# %bb.2:
	movl	%ecx, %ebx
	andl	$-2, %ebx
	xorl	%esi, %esi
	.p2align	4, 0x90
LBB0_3:                                 # =>This Inner Loop Header: Depth=1
	fld	%st(0)
	fmuls	(%edx,%esi,4)
	fadds	(%eax,%esi,4)
	fstps	(%eax,%esi,4)
	fld	%st(0)
	fmuls	4(%edx,%esi,4)
	fadds	4(%eax,%esi,4)
	fstps	4(%eax,%esi,4)
	addl	$2, %esi
	cmpl	%esi, %ebx
	jne	LBB0_3
LBB0_4:
	testb	$1, %cl
	je	LBB0_6
# %bb.5:
	fmuls	(%edx,%esi,4)
	fadds	(%eax,%esi,4)
	fstps	(%eax,%esi,4)
	fldz
LBB0_6:
	fstp	%st(0)
	popl	%esi
	popl	%ebx
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
	pushl	%esi
	movl	24(%esp), %ecx
	flds	12(%esp)
	testl	%ecx, %ecx
	je	LBB1_6
# %bb.1:
	movl	20(%esp), %eax
	movl	16(%esp), %edx
	xorl	%esi, %esi
	cmpl	$1, %ecx
	je	LBB1_4
# %bb.2:
	movl	%ecx, %ebx
	andl	$-2, %ebx
	xorl	%esi, %esi
	.p2align	4, 0x90
LBB1_3:                                 # =>This Inner Loop Header: Depth=1
	fld	%st(0)
	fmuls	(%edx,%esi,4)
	fadds	(%eax,%esi,4)
	fstps	(%eax,%esi,4)
	fld	%st(0)
	fmuls	4(%edx,%esi,4)
	fadds	4(%eax,%esi,4)
	fstps	4(%eax,%esi,4)
	addl	$2, %esi
	cmpl	%esi, %ebx
	jne	LBB1_3
LBB1_4:
	testb	$1, %cl
	je	LBB1_6
# %bb.5:
	fmuls	(%edx,%esi,4)
	fadds	(%eax,%esi,4)
	fstps	(%eax,%esi,4)
	fldz
LBB1_6:
	fstp	%st(0)
	popl	%esi
	popl	%ebx
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
	pushl	%esi
	movl	28(%esp), %ecx
	fldl	12(%esp)
	testl	%ecx, %ecx
	je	LBB2_6
# %bb.1:
	movl	24(%esp), %eax
	movl	20(%esp), %edx
	xorl	%esi, %esi
	cmpl	$1, %ecx
	je	LBB2_4
# %bb.2:
	movl	%ecx, %ebx
	andl	$-2, %ebx
	xorl	%esi, %esi
	.p2align	4, 0x90
LBB2_3:                                 # =>This Inner Loop Header: Depth=1
	fld	%st(0)
	fmull	(%edx,%esi,8)
	faddl	(%eax,%esi,8)
	fstpl	(%eax,%esi,8)
	fld	%st(0)
	fmull	8(%edx,%esi,8)
	faddl	8(%eax,%esi,8)
	fstpl	8(%eax,%esi,8)
	addl	$2, %esi
	cmpl	%esi, %ebx
	jne	LBB2_3
LBB2_4:
	testb	$1, %cl
	je	LBB2_6
# %bb.5:
	fmull	(%edx,%esi,8)
	faddl	(%eax,%esi,8)
	fstpl	(%eax,%esi,8)
	fldz
LBB2_6:
	fstp	%st(0)
	popl	%esi
	popl	%ebx
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
	pushl	%esi
	movl	28(%esp), %ecx
	fldl	12(%esp)
	testl	%ecx, %ecx
	je	LBB3_6
# %bb.1:
	movl	24(%esp), %eax
	movl	20(%esp), %edx
	xorl	%esi, %esi
	cmpl	$1, %ecx
	je	LBB3_4
# %bb.2:
	movl	%ecx, %ebx
	andl	$-2, %ebx
	xorl	%esi, %esi
	.p2align	4, 0x90
LBB3_3:                                 # =>This Inner Loop Header: Depth=1
	fld	%st(0)
	fmull	(%edx,%esi,8)
	faddl	(%eax,%esi,8)
	fstpl	(%eax,%esi,8)
	fld	%st(0)
	fmull	8(%edx,%esi,8)
	faddl	8(%eax,%esi,8)
	fstpl	8(%eax,%esi,8)
	addl	$2, %esi
	cmpl	%esi, %ebx
	jne	LBB3_3
LBB3_4:
	testb	$1, %cl
	je	LBB3_6
# %bb.5:
	fmull	(%edx,%esi,8)
	faddl	(%eax,%esi,8)
	fstpl	(%eax,%esi,8)
	fldz
LBB3_6:
	fstp	%st(0)
	popl	%esi
	popl	%ebx
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
	movl	24(%esp), %edi
	fldz
	testl	%edi, %edi
	je	LBB4_7
# %bb.1:
	fstp	%st(0)
	movl	20(%esp), %esi
	movl	16(%esp), %edx
	movl	%edi, %ecx
	andl	$3, %ecx
	fldz
	xorl	%ebx, %ebx
	cmpl	$4, %edi
	jb	LBB4_4
# %bb.2:
	fstp	%st(0)
	andl	$-4, %edi
	fldz
	xorl	%ebx, %ebx
	.p2align	4, 0x90
LBB4_3:                                 # =>This Inner Loop Header: Depth=1
	flds	(%edx,%ebx,4)
	fmuls	(%esi,%ebx,4)
	flds	4(%edx,%ebx,4)
	fmuls	4(%esi,%ebx,4)
	fxch	%st(1)
	faddp	%st, %st(2)
	flds	8(%edx,%ebx,4)
	fmuls	8(%esi,%ebx,4)
	fxch	%st(1)
	faddp	%st, %st(2)
	flds	12(%edx,%ebx,4)
	fmuls	12(%esi,%ebx,4)
	fxch	%st(1)
	faddp	%st, %st(2)
	faddp	%st, %st(1)
	addl	$4, %ebx
	cmpl	%ebx, %edi
	jne	LBB4_3
LBB4_4:
	testl	%ecx, %ecx
	je	LBB4_7
# %bb.5:
	leal	(%esi,%ebx,4), %esi
	leal	(%edx,%ebx,4), %edx
	xorl	%edi, %edi
	.p2align	4, 0x90
LBB4_6:                                 # =>This Inner Loop Header: Depth=1
	flds	(%edx,%edi,4)
	fmuls	(%esi,%edi,4)
	faddp	%st, %st(1)
	incl	%edi
	cmpl	%edi, %ecx
	jne	LBB4_6
LBB4_7:
	fstps	(%eax)
	popl	%esi
	popl	%edi
	popl	%ebx
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
	movl	24(%esp), %edi
	fldz
	testl	%edi, %edi
	je	LBB5_7
# %bb.1:
	fstp	%st(0)
	movl	20(%esp), %esi
	movl	16(%esp), %edx
	movl	%edi, %ecx
	andl	$3, %ecx
	fldz
	xorl	%ebx, %ebx
	cmpl	$4, %edi
	jb	LBB5_4
# %bb.2:
	fstp	%st(0)
	andl	$-4, %edi
	fldz
	xorl	%ebx, %ebx
	.p2align	4, 0x90
LBB5_3:                                 # =>This Inner Loop Header: Depth=1
	flds	(%edx,%ebx,4)
	fmuls	(%esi,%ebx,4)
	flds	4(%edx,%ebx,4)
	fmuls	4(%esi,%ebx,4)
	fxch	%st(1)
	faddp	%st, %st(2)
	flds	8(%edx,%ebx,4)
	fmuls	8(%esi,%ebx,4)
	fxch	%st(1)
	faddp	%st, %st(2)
	flds	12(%edx,%ebx,4)
	fmuls	12(%esi,%ebx,4)
	fxch	%st(1)
	faddp	%st, %st(2)
	faddp	%st, %st(1)
	addl	$4, %ebx
	cmpl	%ebx, %edi
	jne	LBB5_3
LBB5_4:
	testl	%ecx, %ecx
	je	LBB5_7
# %bb.5:
	leal	(%esi,%ebx,4), %esi
	leal	(%edx,%ebx,4), %edx
	xorl	%edi, %edi
	.p2align	4, 0x90
LBB5_6:                                 # =>This Inner Loop Header: Depth=1
	flds	(%edx,%edi,4)
	fmuls	(%esi,%edi,4)
	faddp	%st, %st(1)
	incl	%edi
	cmpl	%edi, %ecx
	jne	LBB5_6
LBB5_7:
	fstps	(%eax)
	popl	%esi
	popl	%edi
	popl	%ebx
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
