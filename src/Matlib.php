<?php
namespace Rindow\Matlib\FFI;

use Interop\Polite\Math\Matrix\NDArray;
use Interop\Polite\Math\Matrix\LinearBuffer as Buffer;
use InvalidArgumentException;
use RuntimeException;
use LogicException;
use FFI;

class Matlib
{
    use Utils;

    const NO_TRANS = 111;
    const TRANS = 112;
    const CONJ_TRANSTrans = 113;
    const CONJ_NO_TRANSNoTrans = 114;

    const SUCCESS                 =  0;
    const E_MEM_ALLOC_FAILURE     = -101;
    const E_PERM_OUT_OF_RANGE     = -102;
    const E_DUP_AXIS              = -103;
    const E_UNSUPPORTED_DATA_TYPE = -104;
    const E_UNMATCH_IMAGE_BUFFER_SIZE = -105;
    const E_UNMATCH_COLS_BUFFER_SIZE = -106;
    const E_INVALID_SHAPE_OR_PARAM = -107;
    const E_IMAGES_OUT_OF_RANGE   = -108;
    const E_COLS_OUT_OF_RANGE     = -109;

    
    const P_SEQUENTIAL = 0; // Matlib is compiled for sequential use
    const P_THREAD     = 1; // Matlib is compiled using normal threading model
    const P_OPENMP     = 2; // Matlib is compiled using OpenMP threading model

    protected $dtypeToString = [
        NDArray::bool=>'bool',
        NDArray::int8=>'int8',   NDArray::uint8=>'uint8',
        NDArray::int16=>'int16', NDArray::uint16=>'uint16',
        NDArray::int32=>'int32', NDArray::uint32=>'uint32',
        NDArray::int64=>'int64', NDArray::uint64=>'uint64',
        NDArray::float16=>'float16',
        NDArray::float32=>'float32', NDArray::float64=>'float64',
        NDArray::complex64=>'complex64', NDArray::complex128=>'complex128',
    ];

    protected object $ffi;

    public function __construct(FFI $ffi)
    {
        $this->ffi = $ffi;
    }

    public function getNumThreads() : int
    {
        return $this->ffi->rindow_matlib_common_get_num_threads();
    }

    public function getNumProcs() : int
    {
        return $this->ffi->rindow_matlib_common_get_nprocs();
    }

    public function getParallel() : int
    {
        return $this->ffi->rindow_matlib_common_get_parallel();
    }

    public function getVersion() : string
    {
        $string = $this->ffi->rindow_matlib_common_get_version();
        return FFI::string($string);
    }

    /**
     *     sum := sum(X)
     */
    public function sum(
        int $n,
        Buffer $X, int $offsetX, int $incX ) : float
    {
        $this->assert_shape_parameter("n", $n);
        $this->assert_vector_buffer_spec("X", $X,$n,$offsetX,$incX);
        switch ($X->dtype()) {
            case NDArray::float32:{
                $pDataX = $X->addr($offsetX);
                // float rindow_matlib_s_sum(int32_t n,float *x,int32_t incX);
                $result = $this->ffi->rindow_matlib_s_sum($n,$pDataX,$incX);
                break;
            }
            case NDArray::float64:{
                $pDataX = $X->addr($offsetX);
                // double rindow_matlib_d_sum(int32_t n,double *x,int32_t incX);
                $result = $this->ffi->rindow_matlib_d_sum($n,$pDataX,$incX);
                break;
            }
            case NDArray::int8:
            case NDArray::uint8:
            case NDArray::int16:
            case NDArray::uint16:
            case NDArray::int32:
            case NDArray::uint32:
            case NDArray::int64:
            case NDArray::uint64:
            case NDArray::bool: {
                $pDataX = $X->addr($offsetX);
                // int64_t rindow_matlib_i_sum(int32_t dtype, int32_t n,void *x,int32_t incX);
                $result = $this->ffi->rindow_matlib_i_sum($X->dtype(), $n, $pDataX, $incX);
                break;
            }
            default:{
                throw new InvalidArgumentException("Unsupported data type.");
            }
        }
        return $result;
    }

    /**
     *     index := max(X)
     */
    public function imax(
        int $n,
        Buffer $X, int $offsetX, int $incX) : int
    {
        $this->assert_shape_parameter("n", $n);
        $this->assert_vector_buffer_spec("X", $X,$n,$offsetX,$incX);
        switch ($X->dtype()) {
            case NDArray::float32:{
                $pDataX = $X->addr($offsetX);
                $resultIdx = $this->ffi->rindow_matlib_s_imax($n,$pDataX,$incX);
                break;
            }
            case NDArray::float64:{
                $pDataX = $X->addr($offsetX);
                $resultIdx = $this->ffi->rindow_matlib_d_imax($n,$pDataX,$incX);
                break;
            }
            default:{
                if(!$this->is_integer_dtype($X->dtype())) {
                    throw new InvalidArgumentException("Unsupported data type.");
                }
                $pDataX = $X->addr($offsetX);
                $resultIdx = $this->ffi->rindow_matlib_i_imax($X->dtype(), $n, $pDataX, $incX);
                break;
            }
        }
        return $resultIdx;
    }

    /**
     *     index := min(X)
     */
    public function imin(
        int $n,
        Buffer $X, int $offsetX, int $incX) : int
    {
        $this->assert_shape_parameter("n", $n);
        $this->assert_vector_buffer_spec("X", $X,$n,$offsetX,$incX);
        switch ($X->dtype()) {
            case NDArray::float32:{
                $pDataX = $X->addr($offsetX);
                $resultIdx = $this->ffi->rindow_matlib_s_imin($n,$pDataX,$incX);
                break;
            }
            case NDArray::float64:{
                $pDataX = $X->addr($offsetX);
                $resultIdx = $this->ffi->rindow_matlib_d_imin($n,$pDataX,$incX);
                break;
            }
            default:{
                if(!$this->is_integer_dtype($X->dtype())) {
                    throw new InvalidArgumentException("Unsupported data type.");
                }
                $pDataX = $X->addr($offsetX);
                $resultIdx = $this->ffi->rindow_matlib_i_imin($X->dtype(), $n, $pDataX, $incX);
                break;
            }
        }
        return $resultIdx;
    }

    /**
     *     X := a*X + b
     */
    public function increment(
        int $n,
        float $alpha,
        Buffer $X, int $offsetX, int $incX,
        float $beta) : void
    {
        $this->assert_shape_parameter("n", $n);
        $this->assert_vector_buffer_spec("X", $X,$n,$offsetX,$incX);
        switch ($X->dtype()) {
            case NDArray::float32:{
                $pDataX = $X->addr($offsetX);
                $this->ffi->rindow_matlib_s_increment($n, $pDataX, $incX, $alpha, $beta);
                break;
            }
            case NDArray::float64:{
                $pDataX = $X->addr($offsetX);
                $this->ffi->rindow_matlib_d_increment($n, $pDataX, $incX, $alpha, $beta);
                break;
            }
            default:{
                throw new InvalidArgumentException("Unsupported data type.");
            }
        }
    }

    /**
     *     X := 1 / (a*X + b)
     */
    public function reciprocal(
        int $n,
        float $alpha,
        Buffer $X, int $offsetX, int $incX,
        float $beta) : void
    {
        $this->assert_shape_parameter("n", $n);
        $this->assert_vector_buffer_spec("X", $X,$n,$offsetX,$incX);
        switch ($X->dtype()) {
            case NDArray::float32: {
                $pDataX = $X->addr($offsetX);
                $this->ffi->rindow_matlib_s_reciprocal($n, $pDataX, $incX, $alpha, $beta);
                break;
            }
            case NDArray::float64: {
                $pDataX = $X->addr($offsetX);
                $this->ffi->rindow_matlib_d_reciprocal($n, $pDataX, $incX, $alpha, $beta);
                break;
            }
            default: {
                throw new InvalidArgumentException("Unsupported data type.");
            }
        }
    }

    /**
     *     A[m,n] := A[m,n] (A[m,n] >  X[n])
     *     A[m,n] := X[n]   (A[m,n] <= X[n])
     */
    public function maximum(
        int $m,
        int $n,
        Buffer $A, int $offsetA, int $ldA,
        Buffer $X, int $offsetX, int $incX
        ) : void
    {
        $this->assert_shape_parameter("m", $m);
        $this->assert_shape_parameter("n", $n);
        $this->assert_matrix_buffer_spec("A", $A,$m,$n,$offsetA,$ldA);
    
        $this->assert_vector_buffer_spec("X", $X,$n,$offsetX,$incX);
        // Check Buffer X and A
        if($A->dtype()!=$X->dtype()) {
            $types = $this->dtypeToString[$A->dtype()].','.$this->dtypeToString[$X->dtype()];
            throw new InvalidArgumentException("Unmatch data type for A and X: ".$types);
        }
        switch ($A->dtype()) {
            case NDArray::float32: {
                $pDataA = $A->addr($offsetA);
                $pDataX = $X->addr($offsetX);
                $this->ffi->rindow_matlib_s_maximum($m, $n, $pDataA, $ldA, $pDataX, $incX);
                break;
            }
            case NDArray::float64: {
                $pDataA = $A->addr($offsetA);
                $pDataX = $X->addr($offsetX);
                $this->ffi->rindow_matlib_d_maximum($m, $n, $pDataA, $ldA, $pDataX, $incX);
                break;
            }
            default: {
                throw new InvalidArgumentException("Unsupported data type.");
            }
        }
    }

    /**
     *     A[m,n] := A[m,n] (A[m,n] <  X[n])
     *     A[m,n] := X[n]   (A[m,n] >= X[n])
     */
    public function minimum(
        int $m,
        int $n,
        Buffer $A, int $offsetA, int $ldA,
        Buffer $X, int $offsetX, int $incX
        ) : void
    {
        $this->assert_shape_parameter("m", $m);
        $this->assert_shape_parameter("n", $n);
        $this->assert_matrix_buffer_spec("A", $A,$m,$n,$offsetA,$ldA);
    
        $this->assert_vector_buffer_spec("X", $X,$n,$offsetX,$incX);
        // Check Buffer X and A
        if($A->dtype()!=$X->dtype()) {
            $types = $this->dtypeToString[$A->dtype()].','.$this->dtypeToString[$X->dtype()];
            throw new InvalidArgumentException("Unmatch data type for A and X: ".$types);
        }
        switch ($A->dtype()) {
            case NDArray::float32: {
                $pDataA = $A->addr($offsetA);
                $pDataX = $X->addr($offsetX);
                $this->ffi->rindow_matlib_s_minimum($m, $n, $pDataA, $ldA, $pDataX, $incX);
                break;
            }
            case NDArray::float64: {
                $pDataA = $A->addr($offsetA);
                $pDataX = $X->addr($offsetX);
                $this->ffi->rindow_matlib_d_minimum($m, $n, $pDataA, $ldA, $pDataX, $incX);
                break;
            }
            default: {
                throw new InvalidArgumentException("Unsupported data type.");
            }
        }
    }

    /**
     *     A[m,n] := 1 (A[m,n] >  X[n])
     *     A[m,n] := 0 (A[m,n] <= X[n])
     */
    public function greater(
        int $m,
        int $n,
        Buffer $A, int $offsetA, int $ldA,
        Buffer $X, int $offsetX, int $incX
        ) : void
    {
        $this->assert_shape_parameter("m", $m);
        $this->assert_shape_parameter("n", $n);
        $this->assert_matrix_buffer_spec("A", $A,$m,$n,$offsetA,$ldA);
    
        $this->assert_vector_buffer_spec("X", $X,$n,$offsetX,$incX);
        // Check Buffer X and A
        if($A->dtype()!=$X->dtype()) {
            $types = $this->dtypeToString[$A->dtype()].','.$this->dtypeToString[$X->dtype()];
            throw new InvalidArgumentException("Unmatch data type for A and X");
        }
        switch ($A->dtype()) {
            case NDArray::float32: {
                $pDataA = $A->addr($offsetA);
                $pDataX = $X->addr($offsetX);
                $this->ffi->rindow_matlib_s_greater($m, $n, $pDataA, $ldA, $pDataX, $incX);
                break;
            }
            case NDArray::float64: {
                $pDataA = $A->addr($offsetA);
                $pDataX = $X->addr($offsetX);
                $this->ffi->rindow_matlib_d_greater($m, $n, $pDataA, $ldA, $pDataX, $incX);
                break;
            }
            default: {
                throw new InvalidArgumentException("Unsupported data type.");
            }
        }
    }

    /**
     *     A[m,n] := 1 (A[m,n] >= X[n])
     *     A[m,n] := 0 (A[m,n] <  X[n])
     */
    public function greaterEqual(
        int $m,
        int $n,
        Buffer $A, int $offsetA, int $ldA,
        Buffer $X, int $offsetX, int $incX
        ) : void
    {
        $this->assert_shape_parameter("m", $m);
        $this->assert_shape_parameter("n", $n);
        $this->assert_matrix_buffer_spec("A", $A,$m,$n,$offsetA,$ldA);
    
        $this->assert_vector_buffer_spec("X", $X,$n,$offsetX,$incX);
        // Check Buffer X and A
        if($A->dtype()!=$X->dtype()) {
            throw new InvalidArgumentException("Unmatch data type for A and X");
        }
        switch ($A->dtype()) {
            case NDArray::float32: {
                $pDataA = $A->addr($offsetA);
                $pDataX = $X->addr($offsetX);
                $this->ffi->rindow_matlib_s_greater_equal($m, $n, $pDataA, $ldA, $pDataX, $incX);
                break;
            }
            case NDArray::float64: {
                $pDataA = $A->addr($offsetA);
                $pDataX = $X->addr($offsetX);
                $this->ffi->rindow_matlib_d_greater_equal($m, $n, $pDataA, $ldA, $pDataX, $incX);
                break;
            }
            default: {
                throw new InvalidArgumentException("Unsupported data type.");
            }
        }
    }

    /**
     *     A[m,n] := 1 (A[m,n] <  X[n])
     *     A[m,n] := 0 (A[m,n] >= X[n])
     */
    public function less(
        int $m,
        int $n,
        Buffer $A, int $offsetA, int $ldA,
        Buffer $X, int $offsetX, int $incX
        ) : void
    {
        $this->assert_shape_parameter("m", $m);
        $this->assert_shape_parameter("n", $n);
        $this->assert_matrix_buffer_spec("A", $A,$m,$n,$offsetA,$ldA);
    
        $this->assert_vector_buffer_spec("X", $X,$n,$offsetX,$incX);
        // Check Buffer X and A
        if($A->dtype()!=$X->dtype()) {
            throw new InvalidArgumentException("Unmatch data type for A and X");
        }
        switch ($A->dtype()) {
            case NDArray::float32: {
                $pDataA = $A->addr($offsetA);
                $pDataX = $X->addr($offsetX);
                $this->ffi->rindow_matlib_s_less($m, $n, $pDataA, $ldA, $pDataX, $incX);
                break;
            }
            case NDArray::float64: {
                $pDataA = $A->addr($offsetA);
                $pDataX = $X->addr($offsetX);
                $this->ffi->rindow_matlib_d_less($m, $n, $pDataA, $ldA, $pDataX, $incX);
                break;
            }
            default: {
                throw new InvalidArgumentException("Unsupported data type.");
            }
        }
    }

    /**
     *     A[m,n] := 1 (A[m,n] <= X[n])
     *     A[m,n] := 0 (A[m,n] >  X[n])
     */
    public function lessEqual(
        int $m,
        int $n,
        Buffer $A, int $offsetA, int $ldA,
        Buffer $X, int $offsetX, int $incX
        ) : void
    {
        $this->assert_shape_parameter("m", $m);
        $this->assert_shape_parameter("n", $n);
        $this->assert_matrix_buffer_spec("A", $A,$m,$n,$offsetA,$ldA);
    
        $this->assert_vector_buffer_spec("X", $X,$n,$offsetX,$incX);
        // Check Buffer X and A
        if($A->dtype()!=$X->dtype()) {
            $types = $this->dtypeToString[$A->dtype()].','.$this->dtypeToString[$X->dtype()];
            throw new InvalidArgumentException("Unmatch data type for A and X: ".$types);
        }
        switch ($A->dtype()) {
            case NDArray::float32: {
                $pDataA = $A->addr($offsetA);
                $pDataX = $X->addr($offsetX);
                $this->ffi->rindow_matlib_s_less_equal($m, $n, $pDataA, $ldA, $pDataX, $incX);
                break;
            }
            case NDArray::float64: {
                $pDataA = $A->addr($offsetA);
                $pDataX = $X->addr($offsetX);
                $this->ffi->rindow_matlib_d_less_equal($m, $n, $pDataA, $ldA, $pDataX, $incX);
                break;
            }
            default: {
                throw new InvalidArgumentException("Unsupported data type.");
            }
        }
    }

    /**
     *    A(m,n) := X(n) * A(m,n)
     */
    public function multiply(
        bool $trans,
        int $m,
        int $n,
        Buffer $X, int $offsetX, int $incX,
        Buffer $A, int $offsetA, int $ldA
        ) : void
    {
        $this->assert_shape_parameter("m", $m);
        $this->assert_shape_parameter("n", $n);
    
        if($trans) {
            $transCode = self::TRANS;
            $cols = $m;
        } else {
            $transCode = self::NO_TRANS;
            $cols = $n;
        }
    
        // Check Buffer X
        $this->assert_vector_buffer_spec("X", $X,$cols,$offsetX,$incX);
    
        // Check Buffer A
        $this->assert_matrix_buffer_spec("A", $A,$m,$n,$offsetA,$ldA);
    
        // Check Buffer X and A
        if($X->dtype()!=$A->dtype()) {
            $types = $this->dtypeToString[$X->dtype()].','.$this->dtypeToString[$A->dtype()];
            throw new InvalidArgumentException("Unmatch data type for X and A: ".$types);
        }
    
        switch ($X->dtype()) {
            case NDArray::float32: {
                $pDataA = $A->addr($offsetA);
                $pDataX = $X->addr($offsetX);
                $this->ffi->rindow_matlib_s_multiply($transCode, $m, $n, $pDataX, $incX, $pDataA, $ldA);
                break;
            }
            case NDArray::float64: {
                $pDataA = $A->addr($offsetA);
                $pDataX = $X->addr($offsetX);
                $this->ffi->rindow_matlib_d_multiply($transCode, $m, $n, $pDataX, $incX, $pDataA, $ldA);
                break;
            }
            default: {
                throw new InvalidArgumentException("Unsupported data type.");
            }
        }
    }

    /**
     *     A(m,n) := alpha * X(n) + A(m,n)
     */
    public function add(
        bool $trans,
        int $m,
        int $n,
        float $alpha,
        Buffer $X, int $offsetX, int $incX,
        Buffer $A, int $offsetA, int $ldA
        ) : void
    {
        $this->assert_shape_parameter("m", $m);
        $this->assert_shape_parameter("n", $n);

        // Check Buffer X
        if($trans) {
            $transCode = self::TRANS;
            $cols = $m;
        } else {
            $transCode = self::NO_TRANS;
            $cols = $n;
        }
        $this->assert_vector_buffer_spec("X", $X,$cols,$offsetX,$incX);
    
        // Check Buffer A
        $this->assert_matrix_buffer_spec("A", $A,$m,$n,$offsetA,$ldA);
    
        // Check Buffer X and A
        if($X->dtype()!=$A->dtype()) {
            $types = $this->dtypeToString[$X->dtype()].','.$this->dtypeToString[$A->dtype()];
            throw new InvalidArgumentException("Unmatch data type for X and A");
        }
    
        switch ($X->dtype()) {
            case NDArray::float32: {
                $pDataA = $A->addr($offsetA);
                $pDataX = $X->addr($offsetX);
                $this->ffi->rindow_matlib_s_add($transCode, $m, $n, $alpha, $pDataX, $incX, $pDataA, $ldA);
                break;
            }
            case NDArray::float64: {
                $pDataA = $A->addr($offsetA);
                $pDataX = $X->addr($offsetX);
                $this->ffi->rindow_matlib_d_add($transCode, $m, $n, $alpha, $pDataX, $incX, $pDataA, $ldA);
                break;
            }
            default: {
                throw new InvalidArgumentException("Unsupported data type.");
            }
        }
    }

    /**
     *     A(m,n) := X(n)
     */
    public function duplicate(
        bool $trans,
        int $m,
        int $n,
        Buffer $X, int $offsetX, int $incX,
        Buffer $A, int $offsetA, int $ldA
        ) : void
    {
        $this->assert_shape_parameter("m", $m);
        $this->assert_shape_parameter("n", $n);

        // Check Buffer X
        if($trans) {
            $transCode = self::TRANS;
            $cols = $m;
        } else {
            $transCode = self::NO_TRANS;
            $cols = $n;
        }
        $this->assert_vector_buffer_spec("X", $X,$cols,$offsetX,$incX);
    
        // Check Buffer A
        $this->assert_matrix_buffer_spec("A", $A,$m,$n,$offsetA,$ldA);
    
        // Check Buffer X and A
        if($X->dtype()!=$A->dtype()) {
            $types = $this->dtypeToString[$X->dtype()].','.$this->dtypeToString[$A->dtype()];
            throw new InvalidArgumentException("Unmatch data type for X and A");
        }
    
        switch ($X->dtype()) {
            case NDArray::float32: {
                $pDataA = $A->addr($offsetA);
                $pDataX = $X->addr($offsetX);
                $this->ffi->rindow_matlib_s_duplicate($transCode, $m, $n, $pDataX, $incX, $pDataA, $ldA);
                break;
            }
            case NDArray::float64: {
                $pDataA = $A->addr($offsetA);
                $pDataX = $X->addr($offsetX);
                $this->ffi->rindow_matlib_d_duplicate($transCode, $m, $n, $pDataX, $incX, $pDataA, $ldA);
                break;
            }
            default: {
                throw new InvalidArgumentException("Unsupported data type.");
            }
        }
    }

    /**
     *     X := X ^ 2
     */
    public function square(
        int $n,
        Buffer $X, int $offsetX, int $incX
        ) : void
    {
        $this->assert_shape_parameter("n", $n);
        $this->assert_vector_buffer_spec("X", $X,$n,$offsetX,$incX);
        switch ($X->dtype()) {
            case NDArray::float32: {
                $pDataX = $X->addr($offsetX);
                $this->ffi->rindow_matlib_s_square($n, $pDataX, $incX);
                break;
            }
            case NDArray::float64: {
                $pDataX = $X->addr($offsetX);
                $this->ffi->rindow_matlib_d_square($n, $pDataX, $incX);
                break;
            }
            default: {
                throw new InvalidArgumentException("Unsupported data type.");
            }
        }
    }

    /**
     *     X := sqrt(X)
     */
    public function sqrt(
        int $n,
        Buffer $X, int $offsetX, int $incX
        ) : void
    {
        $this->assert_shape_parameter("n", $n);
        $this->assert_vector_buffer_spec("X", $X,$n,$offsetX,$incX);
        switch ($X->dtype()) {
            case NDArray::float32: {
                $pDataX = $X->addr($offsetX);
                $this->ffi->rindow_matlib_s_sqrt($n, $pDataX, $incX);
                break;
            }
            case NDArray::float64: {
                $pDataX = $X->addr($offsetX);
                $this->ffi->rindow_matlib_d_sqrt($n, $pDataX, $incX);
                break;
            }
            default: {
                throw new InvalidArgumentException("Unsupported data type.");
            }
        }
    }

    /**
     *     X := 1 / (a * sqrt(X) + b)
     */
    public function rsqrt(
        int $n,
        float $alpha,
        Buffer $X, int $offsetX, int $incX,
        float $beta) : void
    {
        $this->assert_shape_parameter("n", $n);
        $this->assert_vector_buffer_spec("X", $X,$n,$offsetX,$incX);
        switch ($X->dtype()) {
            case NDArray::float32: {
                $pDataX = $X->addr($offsetX);
                $this->ffi->rindow_matlib_s_rsqrt($n, $alpha, $pDataX, $incX, $beta);
                break;
            }
            case NDArray::float64: {
                $pDataX = $X->addr($offsetX);
                $this->ffi->rindow_matlib_d_rsqrt($n, $alpha, $pDataX, $incX, $beta);
                break;
            }
            default: {
                throw new InvalidArgumentException("Unsupported data type.");
            }
        }
    }

    /**
     *     A(m,n) := A(m,n) ** X(n)
     */
    public function pow(
        bool $trans,
        int $m,
        int $n,
        Buffer $A, int $offsetA, int $ldA,
        Buffer $X, int $offsetX, int $incX,
        ) : void
    {
        $this->assert_shape_parameter("m", $m);
        $this->assert_shape_parameter("n", $n);
        if($trans) {
            $transCode = self::TRANS;
            $cols = $m;
        } else {
            $transCode = self::NO_TRANS;
            $cols = $n;
        }
    
        $this->assert_matrix_buffer_spec("A", $A,$m,$n,$offsetA,$ldA);
    
        $this->assert_vector_buffer_spec("X", $X,$cols,$offsetX,$incX);
        // Check Buffer X and A
        if($A->dtype()!=$X->dtype()) {
            $types = $this->dtypeToString[$A->dtype()].','.$this->dtypeToString[$X->dtype()];
            throw new InvalidArgumentException("Unmatch data type for A and X: ".$types);
        }
        switch ($X->dtype()) {
            case NDArray::float32: {
                $pDataA = $A->addr($offsetA);
                $pDataX = $X->addr($offsetX);
                $this->ffi->rindow_matlib_s_pow($transCode, $m, $n, $pDataA, $ldA, $pDataX, $incX);
                break;
            }
            case NDArray::float64: {
                $pDataA = $A->addr($offsetA);
                $pDataX = $X->addr($offsetX);
                $this->ffi->rindow_matlib_d_pow($transCode, $m, $n, $pDataA, $ldA, $pDataX, $incX);
                break;
            }
            default: {
                throw new InvalidArgumentException("Unsupported data type.");
            }
        }
    }

    /**
     *     X(i) := e ^ X(i)
     */
    public function exp(
        int $n,
        Buffer $X, int $offsetX, int $incX
        ) : void
    {
        $this->assert_shape_parameter("n", $n);
        $this->assert_vector_buffer_spec("X", $X,$n,$offsetX,$incX);
        switch ($X->dtype()) {
            case NDArray::float32: {
                $pDataX = $X->addr($offsetX);
                $this->ffi->rindow_matlib_s_exp($n, $pDataX, $incX);
                break;
            }
            case NDArray::float64: {
                $pDataX = $X->addr($offsetX);
                $this->ffi->rindow_matlib_d_exp($n, $pDataX, $incX);
                break;
            }
            default: {
                throw new InvalidArgumentException("Unsupported data type.");
            }
        }
    }

    /**
     *     X := log(X)
     */
    public function log(
        int $n,
        Buffer $X, int $offsetX, int $incX
        ) : void
    {
        $this->assert_shape_parameter("n", $n);
        $this->assert_vector_buffer_spec("X", $X,$n,$offsetX,$incX);
        switch ($X->dtype()) {
            case NDArray::float32: {
                $pDataX = $X->addr($offsetX);
                $this->ffi->rindow_matlib_s_log($n, $pDataX, $incX);
                break;
            }
            case NDArray::float64: {
                $pDataX = $X->addr($offsetX);
                $this->ffi->rindow_matlib_d_log($n, $pDataX, $incX);
                break;
            }
            default: {
                throw new InvalidArgumentException("Unsupported data type.");
            }
        }
    }

    /**
     *     X := tanh(X)
     */
    public function tanh(
        int $n,
        Buffer $X, int $offsetX, int $incX
        ) : void
    {
        $this->assert_shape_parameter("n", $n);
        $this->assert_vector_buffer_spec("X", $X,$n,$offsetX,$incX);
        switch ($X->dtype()) {
            case NDArray::float32: {
                $pDataX = $X->addr($offsetX);
                $this->ffi->rindow_matlib_s_tanh($n, $pDataX, $incX);
                break;
            }
            case NDArray::float64: {
                $pDataX = $X->addr($offsetX);
                $this->ffi->rindow_matlib_d_tanh($n, $pDataX, $incX);
                break;
            }
            default: {
                throw new InvalidArgumentException("Unsupported data type.");
            }
        }
    }

    /**
     *     X := sin(X)
     */
    public function sin(
        int $n,
        Buffer $X, int $offsetX, int $incX
        ) : void
    {
        $this->assert_shape_parameter("n", $n);
        $this->assert_vector_buffer_spec("X", $X,$n,$offsetX,$incX);
        switch ($X->dtype()) {
            case NDArray::float32: {
                $pDataX = $X->addr($offsetX);
                $this->ffi->rindow_matlib_s_sin($n, $pDataX, $incX);
                break;
            }
            case NDArray::float64: {
                $pDataX = $X->addr($offsetX);
                $this->ffi->rindow_matlib_d_sin($n, $pDataX, $incX);
                break;
            }
            default: {
                throw new InvalidArgumentException("Unsupported data type.");
            }
        }
    }

    /**
     *     X := cos(X)
     */
    public function cos(
        int $n,
        Buffer $X, int $offsetX, int $incX
        ) : void
    {
        $this->assert_shape_parameter("n", $n);
        $this->assert_vector_buffer_spec("X", $X,$n,$offsetX,$incX);
        switch ($X->dtype()) {
            case NDArray::float32: {
                $pDataX = $X->addr($offsetX);
                $this->ffi->rindow_matlib_s_cos($n, $pDataX, $incX);
                break;
            }
            case NDArray::float64: {
                $pDataX = $X->addr($offsetX);
                $this->ffi->rindow_matlib_d_cos($n, $pDataX, $incX);
                break;
            }
            default: {
                throw new InvalidArgumentException("Unsupported data type.");
            }
        }
    }

    /**
     *     X := tan(X)
     */
    public function tan(
        int $n,
        Buffer $X, int $offsetX, int $incX
        ) : void
    {
        $this->assert_shape_parameter("n", $n);
        $this->assert_vector_buffer_spec("X", $X,$n,$offsetX,$incX);
        switch ($X->dtype()) {
            case NDArray::float32: {
                $pDataX = $X->addr($offsetX);
                $this->ffi->rindow_matlib_s_tan($n, $pDataX, $incX);
                break;
            }
            case NDArray::float64: {
                $pDataX = $X->addr($offsetX);
                $this->ffi->rindow_matlib_d_tan($n, $pDataX, $incX);
                break;
            }
            default: {
                throw new InvalidArgumentException("Unsupported data type.");
            }
        }
    }

    /**
     *     X := 0
     */
    public function zeros(
        int $n,
        Buffer $X, int $offsetX, int $incX
        ) : void
    {
        $this->assert_shape_parameter("n", $n);
        $this->assert_vector_buffer_spec("X", $X,$n,$offsetX,$incX);
        switch ($X->dtype()) {
            case NDArray::float32: {
                $pDataX = $X->addr($offsetX);
                $this->ffi->rindow_matlib_s_zeros($n, $pDataX, $incX);
                break;
            }
            case NDArray::float64: {
                $pDataX = $X->addr($offsetX);
                $this->ffi->rindow_matlib_d_zeros($n, $pDataX, $incX);
                break;
            }
            case NDArray::int8:
            case NDArray::uint8:
            case NDArray::int16:
            case NDArray::uint16:
            case NDArray::int32:
            case NDArray::uint32:
            case NDArray::int64:
            case NDArray::uint64:
            case NDArray::complex64:
            case NDArray::complex128:
            case NDArray::bool: {
                $pDataX = $X->addr($offsetX);
                $this->ffi->rindow_matlib_i_zeros($X->dtype(), $n, $pDataX, $incX);
                break;
            }
            default: {
                throw new InvalidArgumentException("Unsupported data type.");
            }
        }
    }

    /**
     *     Y := updateAddOnehot(X,a)
     */
    public function updateAddOnehot(
        int $m,
        int $n,
        float $a,
        Buffer $X, int $offsetX, int $incX,
        Buffer $Y, int $offsetY, int $ldY
        ) : void
    {
        $this->assert_shape_parameter("m", $m);
        $this->assert_shape_parameter("n", $n);
        // Check Buffer X
        $this->assert_vector_buffer_spec("X", $X,$m,$offsetX,$incX);
    
        // Check Buffer Y
        $this->assert_matrix_buffer_spec("Y", $Y,$m,$n,$offsetY,$ldY);
    
        // Check Buffer X
        if($X->dtype()==NDArray::bool) {
            throw new InvalidArgumentException("Data type of BufferX must not be bool");
        }
        if(!$this->is_integer_dtype($X->dtype())) {
            throw new InvalidArgumentException("Unsupported data type of label number.");
        }
    
        switch ($Y->dtype()) {
            case NDArray::float32: {
                $pDataY = $Y->addr($offsetY);
                $pDataX = $X->addr($offsetX);
                if($this->ffi->rindow_matlib_s_onehot($X->dtype(), $m, $n, $pDataX, $incX, $a, $pDataY, $ldY)) {
                    throw new RuntimeException("Label number is out of bounds.");
                }
                break;
            }
            case NDArray::float64: {
                $pDataY = $Y->addr($offsetY);
                $pDataX = $X->addr($offsetX);
                if($this->ffi->rindow_matlib_d_onehot($X->dtype, $m, $n, $pDataX, $incX, $a, $pDataY, $ldY)) {
                    throw new RuntimeException("Label number is out of bounds.");
                }
                break;
            }
            default: {
                throw new InvalidArgumentException("Unsupported data type.");
            }
        }
    }

    public function softmax(
        int $m,
        int $n,
        Buffer $A, int $offsetA, int $ldA) : void
    {
        $this->assert_shape_parameter("m", $m);
        $this->assert_shape_parameter("n", $n);
        $this->assert_matrix_buffer_spec("A", $A,$m,$n,$offsetA,$ldA);
    
        switch ($A->dtype()) {
            case NDArray::float32: {
                $pDataA = $A->addr($offsetA);
                $this->ffi->rindow_matlib_s_softmax($m, $n, $pDataA, $ldA);
                break;
            }
            case NDArray::float64: {
                $pDataA = $A->addr($offsetA);
                $this->ffi->rindow_matlib_d_softmax($m, $n, $pDataA, $ldA);
                break;
            }
            default: {
                throw new InvalidArgumentException("Unsupported data type.");
            }
        }
    }

    /**
     * Y(i) := 1  ( X(i) == Y(i) )
     * Y(i) := 0  ( X(i) != Y(i) )
     */
    public function equal(
        int $n,
        Buffer $X, int $offsetX, int $incX,
        Buffer $Y, int $offsetY, int $incY
        ) : void
    {
        $this->assert_shape_parameter("n", $n);
        // Check Buffer X
        $this->assert_vector_buffer_spec("X", $X,$n,$offsetX,$incX);
    
        // Check Buffer Y
        $this->assert_vector_buffer_spec("Y", $Y,$n,$offsetY,$incY);
    
        // Check Buffer X and Y
        if($X->dtype()!=$Y->dtype()) {
            $types = $this->dtypeToString[$X->dtype()].','.$this->dtypeToString[$Y->dtype()];
            throw new InvalidArgumentException("Unmatch data type for X and Y: ".$types);
        }
    
        switch ($X->dtype()) {
            case NDArray::float32: {
                $pDataX = $X->addr($offsetX);
                $pDataY = $Y->addr($offsetY);
                $this->ffi->rindow_matlib_s_equal($n, $pDataX, $incX, $pDataY, $incY);
                break;
            }
            case NDArray::float64: {
                $pDataX = $X->addr($offsetX);
                $pDataY = $Y->addr($offsetY);
                $this->ffi->rindow_matlib_d_equal($n, $pDataX, $incX, $pDataY, $incY);
                break;
            }
            default: {
                if(!$this->is_integer_dtype($X->dtype())&&
                    !($X->dtype()==NDArray::bool)) {
                        throw new InvalidArgumentException("Unsupported data type.");
                }
                $pDataX = $X->addr($offsetX);
                $pDataY = $Y->addr($offsetY);
                $this->ffi->rindow_matlib_i_equal($X->dtype(), $n, $pDataX, $incX, $pDataY, $incY);
                break;
            }
        }
    }

    /**
     * Y(i) := 1  ( X(i) != Y(i) )
     * Y(i) := 0  ( X(i) == Y(i) )
     */
    public function notEqual(
        int $n,
        Buffer $X, int $offsetX, int $incX,
        Buffer $Y, int $offsetY, int $incY
        ) : void
    {
        $this->assert_shape_parameter("n", $n);
        // Check Buffer X
        $this->assert_vector_buffer_spec("X", $X,$n,$offsetX,$incX);
    
        // Check Buffer Y
        $this->assert_vector_buffer_spec("Y", $Y,$n,$offsetY,$incY);
    
        // Check Buffer X and Y
        if($X->dtype()!=$Y->dtype()) {
            $types = $this->dtypeToString[$X->dtype()].','.$this->dtypeToString[$Y->dtype()];
            throw new InvalidArgumentException("Unmatch data type for X and Y");
        }
    
        switch ($X->dtype()) {
            case NDArray::float32: {
                $pDataX = $X->addr($offsetX);
                $pDataY = $Y->addr($offsetY);
                $this->ffi->rindow_matlib_s_notequal($n, $pDataX, $incX, $pDataY, $incY);
                break;
            }
            case NDArray::float64: {
                $pDataX = $X->addr($offsetX);
                $pDataY = $Y->addr($offsetY);
                $this->ffi->rindow_matlib_d_notequal($n, $pDataX, $incX, $pDataY, $incY);
                break;
            }
            default: {
                if(!$this->is_integer_dtype($X->dtype())&&
                    !($X->dtype()==NDArray::bool)) {
                        throw new InvalidArgumentException("Unsupported data type.");
                }
                $pDataX = $X->addr($offsetX);
                $pDataY = $Y->addr($offsetY);
                $this->ffi->rindow_matlib_i_notequal($X->dtype(), $n, $pDataX, $incX, $pDataY, $incY);
                break;
            }
        }
    }

    /**
     * X(i) := 1  ( X(i) == 0 )
     * X(i) := 0  ( X(i) != 0 )
     */
    public function not(
        int $n,
        Buffer $X, int $offsetX, int $incX,
        ) : void
    {
        $this->assert_shape_parameter("n", $n);
        // Check Buffer X
        $this->assert_vector_buffer_spec("X", $X,$n,$offsetX,$incX);
    
        switch ($X->dtype()) {
            case NDArray::float32: {
                $pDataX = $X->addr($offsetX);
                $this->ffi->rindow_matlib_s_not($n, $pDataX, $incX);
                break;
            }
            case NDArray::float64: {
                $pDataX = $X->addr($offsetX);
                $this->ffi->rindow_matlib_d_not($n, $pDataX, $incX);
                break;
            }
            default: {
                if(!$this->is_integer_dtype($X->dtype())&&
                    !($X->dtype()==NDArray::bool)) {
                        throw new InvalidArgumentException("Unsupported data type.");
                }
                $pDataX = $X->addr($offsetX);
                $this->ffi->rindow_matlib_i_not($X->dtype(), $n, $pDataX, $incX);
                break;
            }
        }
    }

    public function astype(
        int $n,
        int $dtype,
        Buffer $X, int $offsetX, int $incX,
        Buffer $Y, int $offsetY, int $incY
        ) : void
    {
        $this->assert_shape_parameter("n", $n);
        $this->assert_vector_buffer_spec("X", $X,$n,$offsetX,$incX);
        $this->assert_vector_buffer_spec("Y", $Y,$n,$offsetY,$incY);
        // Check dtype and Buffer Y
        if($dtype!=$Y->dtype()) {
            throw new InvalidArgumentException("Unmatch data type for Y");
        }
    
        $pDataX = $X->addr($offsetX);
        $pDataY = $Y->addr($offsetY);

        if($this->ffi->rindow_matlib_astype($n, $X->dtype(), $pDataX, $incX, $Y->dtype(), $pDataY, $incY)) {
            throw new InvalidArgumentException("Unsupported data type of X or Y.");
        }
    }

    public function matrixcopy(
        bool $trans,
        int $m,
        int $n,
        float $alpha,
        Buffer $A, int $offsetA, int $ldA,
        Buffer $B, int $offsetB, int $ldB)
    {
        $this->assert_shape_parameter("m", $m);
        $this->assert_shape_parameter("n", $n);
        if($trans) {
            $transCode = self::TRANS;
            $cols = $m;
        } else {
            $transCode = self::NO_TRANS;
            $cols = $n;
        }
        // Check Buffer A
        $this->assert_matrix_buffer_spec("A", $A,$m,$n,$offsetA,$ldA);
    
        if(!$trans) {
            $rows = $m;
            $cols = $n;
        } else {
            $rows = $n;
            $cols = $m;
        }
        // Check Buffer B
        $this->assert_matrix_buffer_spec("B", $B,$rows,$cols,$offsetB,$ldB);
    
        if($A->dtype()!=$B->dtype()) {
            $types = $this->dtypeToString[$A->dtype()].','.$this->dtypeToString[$B->dtype()];
            throw new InvalidArgumentException("Unmatch data type A and B: ".$types);
        }
    
        switch ($A->dtype()) {
            case NDArray::float32: {
                $pDataA = $A->addr($offsetA);
                $pDataB = $B->addr($offsetB);
                $this->ffi->rindow_matlib_s_matrixcopy($transCode, $m, $n, $alpha, $pDataA, $ldA, $pDataB, $ldB);
                break;
            }
            case NDArray::float64: {
                $pDataA = $A->addr($offsetA);
                $pDataB = $B->addr($offsetB);
                $this->ffi->rindow_matlib_d_matrixcopy($transCode, $m, $n, $alpha, $pDataA, $ldA, $pDataB, $ldB);
                break;
            }
            default: {
                throw new InvalidArgumentException("Unsupported data type of A.");
            }
        }
    }

    public function imagecopy(
        int $height,
        int $width,
        int $channels,
        Buffer $A, int $offsetA,
        Buffer $B, int $offsetB,
        bool $channelsFirst,
        int $heightShift,
        int $widthShift,
        bool $verticalFlip,
        bool $horizontalFlip,
        bool $rgbFlip
        )
    {
        if($height<1) {
            throw new InvalidArgumentException("height must be greater then 0");
        }
        if($width<1) {
            throw new InvalidArgumentException("width must be greater then 0");
        }
        if($channels<1) {
            throw new InvalidArgumentException("channels must be greater then 0");
        }
        // Check Buffer A
        if($A->count() < $height*$width*$channels+$offsetA) {
            throw new InvalidArgumentException("Matrix specification too large for bufferA");
        }
        // Check Buffer B
        if($B->count() < $height*$width*$channels+$offsetB) {
            throw new InvalidArgumentException("Matrix specification too large for bufferB");
        }
    
        if($A->dtype()!=$B->dtype()) {
            $types = $this->dtypeToString[$A->dtype()].','.$this->dtypeToString[$B->dtype()];
            throw new InvalidArgumentException("Unmatch data type A and B: ".$types);
        }
    
        if($channelsFirst) {
            $ldC = $width*$height;
            $ldY = $width;
            $ldX = 1;
        } else {
            $ldY = $width*$channels;
            $ldX = $channels;
            $ldC = 1;
        }
        //$directionY = 1;
        //$directionX = 1;
        //$biasY = 0;
        //$biasX = 0;
        //if($verticalFlip) {
        //    $directionY = -$directionY;
        //    $biasY = $height-1;
        //}
        //if($horizontalFlip) {
        //    $directionX = -$directionX;
        //    $biasX = $width-1;
        //}
        //$biasY -= $heightShift*$directionY;
        //$biasX -= $widthShift*$directionX;
    
        switch ($A->dtype()) {
            case NDArray::float32: {
                $pDataA = $A->addr($offsetA);
                $pDataB = $B->addr($offsetB);
                $this->ffi->rindow_matlib_s_imagecopy($height,$width,$channels,$pDataA,$pDataB,
                    $channelsFirst,$heightShift,$widthShift,$verticalFlip,$horizontalFlip,$rgbFlip);
                break;
            }
            case NDArray::float64: {
                $pDataA = $A->addr($offsetA);
                $pDataB = $B->addr($offsetB);
                $this->ffi->rindow_matlib_d_imagecopy($height,$width,$channels,$pDataA,$pDataB,
                    $channelsFirst,$heightShift,$widthShift,$verticalFlip,$horizontalFlip,$rgbFlip);
                break;
            }
            case NDArray::uint8: {
                $pDataA = $A->addr($offsetA);
                $pDataB = $B->addr($offsetB);
                $this->ffi->rindow_matlib_i8_imagecopy($height,$width,$channels,$pDataA,$pDataB,
                    $channelsFirst,$heightShift,$widthShift,$verticalFlip,$horizontalFlip,$rgbFlip);
                break;
            }
            default: {
                throw new InvalidArgumentException("Unsupported data type of A.");
            }
        }
    }

    public function fill(
        int $n,
        Buffer $V, int $offsetV,
        Buffer $X, int $offsetX, int $incX)
    {
        $this->assert_shape_parameter("n", $n);
        // Check Buffer V
        if($offsetV >= $V->count()) {
            throw new InvalidArgumentException("value buffer size is too small");
        }
        // Check Buffer X
        $this->assert_vector_buffer_spec("X", $X,$n,$offsetX,$incX);
    
        if($V->dtype()!=$X->dtype()) {
            throw new InvalidArgumentException("Unmatch data type X and value");
        }
    
        $pDataV = $V->addr($offsetV);
        $pDataX = $X->addr($offsetX);
        $this->ffi->rindow_matlib_fill($X->dtype(), $n, $pDataV, $pDataX, $incX);
    }

    public function nan2num(
        int $n,
        Buffer $X, int $offsetX, int $incX,
        float $alpha
        ) : void
    {
        $this->assert_shape_parameter("n", $n);
        $this->assert_vector_buffer_spec("X", $X,$n,$offsetX,$incX);
        switch ($X->dtype()) {
            case NDArray::float32: {
                $pDataX = $X->addr($offsetX);
                $this->ffi->rindow_matlib_s_nan2num($n, $pDataX, $incX, $alpha);
                break;
            }
            case NDArray::float64: {
                $pDataX = $X->addr($offsetX);
                $this->ffi->rindow_matlib_d_nan2num($n, $pDataX, $incX, $alpha);
                break;
            }
            default: {
                throw new InvalidArgumentException("Unsupported data type.");
            }
        }
    }

    public function isnan(
        int $n,
        Buffer $X, int $offsetX, int $incX
        ) : void
    {
        $this->assert_shape_parameter("n", $n);
        $this->assert_vector_buffer_spec("X", $X,$n,$offsetX,$incX);
        switch ($X->dtype()) {
            case NDArray::float32: {
                $pDataX = $X->addr($offsetX);
                $this->ffi->rindow_matlib_s_isnan($n, $pDataX, $incX);
                break;
            }
            case NDArray::float64: {
                $pDataX = $X->addr($offsetX);
                $this->ffi->rindow_matlib_d_isnan($n, $pDataX, $incX);
                break;
            }
            default: {
                throw new InvalidArgumentException("Unsupported data type.");
            }
        }
    }

    public function searchsorted(
        int $m,
        int $n,
        Buffer $A, int $offsetA, int $ldA, // float
        Buffer $X, int $offsetX, int $incX, // float
        bool $right,
        Buffer $Y, int $offsetY, int $incY // int
        ) : void
    {
        $this->assert_shape_parameter("m", $m);
        $this->assert_shape_parameter("n", $n);
        // Check Buffer A
        if($offsetA<0) {
            throw new InvalidArgumentException("Argument offsetA must be greater than or equals 0.");
        }
        if($ldA<0) {
            throw new InvalidArgumentException("Argument ldA must be greater than or equals 0.");
        }
        if($offsetA+($m-1)*$ldA+($n-1) >= count($A)) {
            throw new InvalidArgumentException("Matrix specification too large for bufferA.");
        }

        // Check Buffer X
        $this->assert_vector_buffer_spec("X", $X,$m,$offsetX,$incX);
    
        // Check Buffer Y
        $this->assert_vector_buffer_spec("Y", $Y,$m,$offsetY,$incY);
    
        // Check Buffer A and X
        if($A->dtype()!=$X->dtype()) {
            $types = $this->dtypeToString[$A->dtype()].','.$this->dtypeToString[$X->dtype()];
            throw new InvalidArgumentException("Unmatch data type for A and X: ".$types);
        }
    
        switch ($A->dtype()) {
            case NDArray::float32: {
                $pDataA = $A->addr($offsetA);
                $pDataX = $X->addr($offsetX);
                $pDataY = $Y->addr($offsetY);
                $this->ffi->rindow_matlib_s_searchsorted($m,$n,$pDataA,$ldA,$pDataX,$incX,$right,$Y->dtype(),$pDataY,$incY);
                break;
            }
            case NDArray::float64: {
                $pDataA = $A->addr($offsetA);
                $pDataX = $X->addr($offsetX);
                $pDataY = $Y->addr($offsetY);
                $this->ffi->rindow_matlib_d_searchsorted($m,$n,$pDataA,$ldA,$pDataX,$incX,$right,$Y->dtype(),$pDataY,$incY);
                break;
            }
            default: {
                throw new InvalidArgumentException("Unsupported data type.");
            }
        }
    }

    public function cumsum(
        int $n,
        Buffer $X, int $offsetX, int $incX, // float
        bool $exclusive,
        bool $reverse,
        Buffer $Y, int $offsetY, int $incY // int
        ) : void
    {
        $this->assert_shape_parameter("n", $n);
    
        // Check Buffer X
        $this->assert_vector_buffer_spec("X", $X,$n,$offsetX,$incX);
    
        // Check Buffer Y
        $this->assert_vector_buffer_spec("Y", $Y,$n,$offsetY,$incY);
    
        // Check Buffer A and X
        if($X->dtype()!=$Y->dtype()) {
            $types = $this->dtypeToString[$X->dtype()].','.$this->dtypeToString[$Y->dtype()];
            throw new InvalidArgumentException("Unmatch data type for X and Y: ".$types);
        }
    
        switch ($X->dtype()) {
            case NDArray::float32: {
                $pDataX = $X->addr($offsetX);
                $pDataY = $Y->addr($offsetY);
                $this->ffi->rindow_matlib_s_cumsum($n,$pDataX,$incX,$exclusive,$reverse,$pDataY,$incY);
                break;
            }
            case NDArray::float64: {
                $pDataX = $X->addr($offsetX);
                $pDataY = $Y->addr($offsetY);
                $this->ffi->rindow_matlib_d_cumsum($n,$pDataX,$incX,$exclusive,$reverse,$pDataY,$incY);
                break;
            }
            default: {
                throw new InvalidArgumentException("Unsupported data type.");
            }
        }
    }

    public function transpose(
        Buffer $sourceShape,
        Buffer $perm,
        Buffer $A, int $offsetA,
        Buffer $B, int $offsetB, 
        ) : void
    {
        // Check Buffer Shape
        $ndim = $sourceShape->count();
        if($ndim<=0) {
            throw new DomainException("ndim must be greater than 0.");
        }
        if($sourceShape->dtype()!=NDArray::int32) {
            throw new InvalidArgumentException("data type of shape buffer must be int32.");
        }
    
        $size = 1;
        for($i=0;$i<$ndim;$i++) {
            if($sourceShape[$i]<=0) {
                throw new InvalidArgumentException("shape values must be greater than 0.");
            }
            $size *= $sourceShape[$i];
        }
    
        // Check Buffer perm
        if($ndim != $perm->count()) {
            throw new InvalidArgumentException("matrix shape and perm must be same size.");
        }
        if($perm->dtype()!=NDArray::int32) {
            throw new InvalidArgumentException("data type of perm buffer must be int32.");
        }
    
        // Check Buffer A
        $this->assert_vector_buffer_spec("A", $A,$size,$offsetA,1);
    
        // Check Buffer B
        $this->assert_vector_buffer_spec("B", $B,$size,$offsetB,1);
    
        // Check Buffer A and B
        if($A->dtype()!=$B->dtype()) {
            $types = $this->dtypeToString[$A->dtype()].','.$this->dtypeToString[$B->dtype()];
            throw new InvalidArgumentException("Unmatch data type for A and B.");
        }
    
        $shapevals=$sourceShape->addr(0);
        $permvals=$perm->addr(0);
        switch ($A->dtype()) {
            case NDArray::float32: {
                $pDataA = $A->addr($offsetA);
                $pDataB = $B->addr($offsetB);
                $status = $this->ffi->rindow_matlib_s_transpose($ndim, $shapevals, $permvals, $pDataA, $pDataB);
                break;
            }
            case NDArray::float64: {
                $pDataA = $A->addr($offsetA);
                $pDataB = $B->addr($offsetB);
                $status = $this->ffi->rindow_matlib_d_transpose($ndim, $shapevals, $permvals, $pDataA, $pDataB);
                break;
            }
            case NDArray::bool:
            case NDArray::int8:
            case NDArray::uint8:
            case NDArray::int16:
            case NDArray::uint16:
            case NDArray::int32:
            case NDArray::uint32:
            case NDArray::int64:
            case NDArray::uint64: {
                $pDataA = $A->addr($offsetA);
                $pDataB = $B->addr($offsetB);
                $status = $this->ffi->rindow_matlib_i_transpose($A->dtype(), $ndim, $shapevals, $permvals, $pDataA, $pDataB);
                break;
            }
            default: {
                throw new InvalidArgumentException("Unsupported data type.");
            }
        }

        if($status==self::SUCCESS) {
            return;
        }
        switch($status) {
            case self::E_MEM_ALLOC_FAILURE: {
                throw new RuntimeException("memory allocation failure");
            }
            case self::E_PERM_OUT_OF_RANGE: {
                throw new InvalidArgumentException("perm contained an out-of-bounds axis");
            }
            case self::E_DUP_AXIS: {
                throw new InvalidArgumentException("Perm contained duplicate axis");
            }
            default: {
                throw new RuntimeException("Unknown error.");
            }
        }
    }

    public function bandpart(
        int $m,
        int $n,
        int $k,
        Buffer $A, int $offsetA,
        int $lower,
        int $upper,
        ) : void
    {
        $this->assert_shape_parameter("m", $m);
        $this->assert_shape_parameter("n", $n);
        $this->assert_shape_parameter("k", $k);
    
        // Check Buffer A
        $this->assert_vector_buffer_spec("A", $A,$m*$n*$k,$offsetA,1);
    
        switch ($A->dtype()) {
            case NDArray::float32: {
                $pDataA = $A->addr($offsetA);
                $this->ffi->rindow_matlib_s_bandpart($m,$n,$k,$pDataA,$lower,$upper);
                break;
            }
            case NDArray::float64: {
                $pDataA = $A->addr($offsetA);
                $this->ffi->rindow_matlib_d_bandpart($m,$n,$k,$pDataA,$lower,$upper);
                break;
            }
            default: {
                throw new InvalidArgumentException("Unsupported data type.");
            }
        }
    }

    /**
    *      B(n,k) := A(X(n),k)
    */
    public function gather(
        bool $reverse,
        bool $addMode,
        int $n,
        int $k,
        int $numClass,
        Buffer $X, int $offsetX,
        Buffer $A, int $offsetA,
        Buffer $B, int $offsetB
        ) : void
    {
        $this->assert_shape_parameter("n", $n);
        $this->assert_shape_parameter("k", $k);
        if($numClass<=0) {
            throw new InvalidArgumentException("Argument numClass must be greater than or equal 0.");
        }
    
        // Check Buffer X
        if($offsetX<0) {
            throw new InvalidArgumentException("Argument offsetX must be greater than or equal 0.");
        }
        if($offsetX+$n > $X->count()) {
            throw new InvalidArgumentException("Matrix X specification too large for buffer.");
        }
    
        // Check Buffer A
        if($offsetA<0) {
            throw new InvalidArgumentException("Argument offsetA must be greater than or equal 0.");
        }
        if($offsetA+$numClass*$k > $A->count()) {
            throw new InvalidArgumentException("Matrix A specification too large for buffer.");
        }
    
        // Check Buffer B
        if($offsetB<0) {
            throw new InvalidArgumentException("Argument offsetB must be greater than or equal 0.");
        }
        if($offsetB+$n*$k > $B->count()) {
            throw new InvalidArgumentException("Matrix B specification too large for buffer.");
        }
    
        // Check Buffer A and Y
        if($A->dtype()!=$B->dtype()) {
            $types = $this->dtypeToString[$A->dtype()].','.$this->dtypeToString[$B->dtype()];
            throw new InvalidArgumentException("Unmatch data type for A and B: ".$types);
        }
        if($X->dtype()==NDArray::bool) {
            throw new InvalidArgumentException("Data type of BufferX must not be bool");
        }
    
        switch ($A->dtype()) {
            case NDArray::float32: {
                $pDataX = $X->addr($offsetX);
                $pDataA = $A->addr($offsetA);
                $pDataB = $B->addr($offsetB);
                $errcode = $this->ffi->rindow_matlib_s_gather($reverse,$addMode,$n,$k,$numClass,$X->dtype(),$pDataX,$pDataA,$pDataB);
                if($errcode) {
                    if($errcode == self::E_UNSUPPORTED_DATA_TYPE) {
                        throw new InvalidArgumentException("Unsupported data type of label number.");
                    } else if($errcode == self::E_PERM_OUT_OF_RANGE) {
                        throw new RuntimeException("Label number is out of bounds.");
                    } else {
                        throw new LogicException(sprintf("Unknown error.(%d)", $errcode));
                    }
                }
                break;
            }
            case NDArray::float64: {
                $pDataX = $X->addr($offsetX);
                $pDataA = $A->addr($offsetA);
                $pDataB = $B->addr($offsetB);
                $errcode = $this->ffi->rindow_matlib_d_gather($reverse,$addMode,$n,$k,$numClass,$X->dtype(),$pDataX,$pDataA,$pDataB);
                if($errcode) {
                    if($errcode == self::E_UNSUPPORTED_DATA_TYPE) {
                        throw new InvalidArgumentException("Unsupported data type of label number.");
                    } else if($errcode == self::E_PERM_OUT_OF_RANGE) {
                        throw new RuntimeException("Label number is out of bounds.");
                    } else {
                        throw new LogicException(sprintf("Unknown error.(%d)", $errcode));
                    }
                }
                break;
            }
            default: {
                if(!$this->is_integer_dtype($A->dtype())&&NDArray::bool!=$A->dtype()) {
                        throw new InvalidArgumentException("Unsupported data type.");
                }
                $pDataX = $X->addr($offsetX);
                $pDataA = $A->addr($offsetA);
                $pDataB = $B->addr($offsetB);
                $errcode = $this->ffi->rindow_matlib_i_gather($reverse,$addMode,$n,$k,$numClass,$X->dtype(),$pDataX,$A->dtype(),$pDataA,$pDataB);
                if($errcode) {
                    if($errcode == self::E_UNSUPPORTED_DATA_TYPE) {
                        throw new InvalidArgumentException("Unsupported data type.");
                    } else if($errcode == self::E_PERM_OUT_OF_RANGE) {
                        throw new RuntimeException("Label number is out of bounds.");
                    } else {
                        throw new LogicException(sprintf("Unknown error.(%d)", $errcode));
                    }
                }
                break;
            }
        }
    }

    /**
    *      B(m,n) := A(m,X(m,n))
    */
    public function reduceGather(
        bool $reverse,
        bool $addMode,
        int $m,
        int $n,
        int $numClass,
        Buffer $X, int $offsetX,
        Buffer $A, int $offsetA,
        Buffer $B, int $offsetB
        ) : void
    {
        $this->assert_shape_parameter("m", $m);
        $this->assert_shape_parameter("n", $n);
        if($numClass<=0) {
            throw new InvalidArgumentException("Argument numClass must be greater than or equal 0.");
        }
        // Check Buffer X
        if($offsetX<0) {
            throw new InvalidArgumentException("Argument offsetX must be greater than or equal 0.");
        }
        if($offsetX+$m*$n > $X->count()) {
            throw new InvalidArgumentException("Matrix X specification too large for buffer.");
        }
    
        // Check Buffer A
        if($offsetA<0) {
            throw new InvalidArgumentException("Argument offsetA must be greater than or equal 0.");
        }
        if($offsetA+$m*$numClass > $A->count()) {
            throw new InvalidArgumentException("Matrix A specification too large for buffer.");
        }
    
        // Check Buffer B
        if($offsetB<0) {
            throw new InvalidArgumentException("Argument offsetB must be greater than or equal 0.");
        }
        if($offsetB+$m*$n > $B->count()) {
            throw new InvalidArgumentException("Matrix B specification too large for buffer.");
        }
    
        // Check Buffer A and Y
        if($A->dtype()!=$B->dtype()) {
            $types = $this->dtypeToString[$A->dtype()].','.$this->dtypeToString[$B->dtype()];
            throw new InvalidArgumentException("Unmatch data type for A and B: ".$types);
        }
        if($X->dtype()==NDArray::bool) {
            throw new InvalidArgumentException("Data type of BufferX must not be bool");
        }
    
        switch ($A->dtype()) {
            case NDArray::float32: {
                $pDataX = $X->addr($offsetX);
                $pDataA = $A->addr($offsetA);
                $pDataB = $B->addr($offsetB);
                $errcode = $this->ffi->rindow_matlib_s_reducegather($reverse,$addMode,$m,$n,$numClass,$X->dtype(),$pDataX,$pDataA,$pDataB);
                if($errcode) {
                    if($errcode == self::E_UNSUPPORTED_DATA_TYPE) {
                        throw new InvalidArgumentException("Unsupported data type of label number.");
                    } else if($errcode == self::E_PERM_OUT_OF_RANGE) {
                        throw new RuntimeException("Label number is out of bounds.");
                    } else {
                        throw new InvalidArgumentException(sprintf("Unknown error.(%d)", $errcode));
                    }
                }
                break;
            }
            case NDArray::float64: {
                $pDataX = $X->addr($offsetX);
                $pDataA = $A->addr($offsetA);
                $pDataB = $B->addr($offsetB);
                $errcode = $this->ffi->rindow_matlib_d_reducegather($reverse,$addMode,$m,$n,$numClass,$X->dtype(),$pDataX,$pDataA,$pDataB);
                if($errcode) {
                    if($errcode == self::E_UNSUPPORTED_DATA_TYPE) {
                        throw new InvalidArgumentException("Unsupported data type of label number.");
                    } else if($errcode == self::E_PERM_OUT_OF_RANGE) {
                        throw new RuntimeException("Label number is out of bounds.");
                    } else {
                        throw new InvalidArgumentException(sprintf("Unknown error.(%d)", $errcode));
                    }
                }
                break;
            }
            default: {
                if(!$this->is_integer_dtype($A->dtype())&&
                    NDArray::bool!=$A->dtype()) {
                        throw new InvalidArgumentException("Unsupported data type.");
                }
                $pDataX = $X->addr($offsetX);
                $pDataA = $A->addr($offsetA);
                $pDataB = $B->addr($offsetB);
                $errcode = $this->ffi->rindow_matlib_i_reducegather($reverse,$addMode,$m,$n,$numClass,$X->dtype(),$pDataX,$A->dtype(),$pDataA,$pDataB);
                if($errcode) {
                    if($errcode == self::E_UNSUPPORTED_DATA_TYPE) {
                        throw new InvalidArgumentException("Unsupported data type.");
                    } else if($errcode == self::E_PERM_OUT_OF_RANGE) {
                        throw new RuntimeException("Label number is out of bounds.");
                    } else {
                        throw new InvalidArgumentException(sprintf("Unknown error.(%d)", $errcode));
                    }
                }
                break;
            }
        }
    }

    /**
    */
    public function slice(
        bool $reverse,
        bool $addMode,
        int $m,
        int $n,
        int $k,
        int $size,
        Buffer $A, int $offsetA, int $incA,
        Buffer $Y, int $offsetY, int $incY,
        int $startAxis0,
        int $sizeAxis0,
        int $startAxis1,
        int $sizeAxis1,
        int $startAxis2,
        int $sizeAxis2
        )
    {
        $this->assert_shape_parameter("m", $m);
        $this->assert_shape_parameter("n", $n);
        $this->assert_shape_parameter("k", $k);
        if($size<=0){
            throw new InvalidArgumentException("Argument size must be greater than or equal 0.");
        }
        if($startAxis0<0){
            throw new InvalidArgumentException("Argument startAxis0 must be greater than or equal 0.");
        }
        if($sizeAxis0<=0){
            throw new InvalidArgumentException("Argument sizeAxis0 must be greater than 0.");
        }
        if($startAxis1<0){
            throw new InvalidArgumentException("Argument startAxis1 must be greater than or equal 0.");
        }
        if($sizeAxis1<=0){
            throw new InvalidArgumentException("Argument sizeAxis1 must be greater than 0.");
        }
        if($startAxis2<0){
            throw new InvalidArgumentException("Argument startAxis2 must be greater than or equal 0.");
        }
        if($sizeAxis2<=0){
            throw new InvalidArgumentException("Argument sizeAxis2 must be greater than 0.");
        }
        // Check Buffer A
        if($m*$n*$k*$size*$incA+$offsetA > $A->count()) {
            throw new InvalidArgumentException("Unmatch BufferA size and m,n,k,size");
        }
        // Check Buffer Y
        if($sizeAxis0*$sizeAxis1*$size*$incY+$offsetY > $Y->count()) {
            throw new InvalidArgumentException("BufferY size is too small");
        }
    
        if($startAxis0>=$m||
            $sizeAxis0+$startAxis0>$m){
            throw new InvalidArgumentException("Axis0 range is too large for source array.");
        }
        if($startAxis1>=$n||
            $sizeAxis1+$startAxis1>$n){
            throw new InvalidArgumentException("Axis1 range is too large for source array.");
        }
        if($startAxis2>=$k||
            $sizeAxis2+$startAxis2>$k){
            throw new InvalidArgumentException("Axis2 range is too large for source array.");
        }
        if($A->dtype() != $Y->dtype()) {
            $types = $this->dtypeToString[$A->dtype()].','.$this->dtypeToString[$Y->dtype()];
            throw new InvalidArgumentException("Unmatch data type: ".$types);
        }
    
        switch ($A->dtype()) {
            case NDArray::float32: {
                $pDataA = $A->addr($offsetA);
                $pDataY = $Y->addr($offsetY);
                $this->ffi->rindow_matlib_s_slice($reverse,$addMode,$m,$n,$k,$size,$pDataA,$incA,$pDataY,$incY,$startAxis0,$sizeAxis0,$startAxis1,$sizeAxis1,$startAxis2,$sizeAxis2);
                break;
            }
            case NDArray::float64: {
                $pDataA = $A->addr($offsetA);
                $pDataY = $Y->addr($offsetY);
                $this->ffi->rindow_matlib_d_slice($reverse,$addMode,$m,$n,$k,$size,$pDataA,$incA,$pDataY,$incY,$startAxis0,$sizeAxis0,$startAxis1,$sizeAxis1,$startAxis2,$sizeAxis2);
                break;
            }
            default:{
                if(!$this->is_integer_dtype($A->dtype())&&
                    NDArray::bool!=$A->dtype()) {
                    throw new InvalidArgumentException("Unsupported data type.");
                }
                $pDataA = $A->addr($offsetA);
                $pDataY = $Y->addr($offsetY);
                $this->ffi->rindow_matlib_i_slice($reverse,$addMode,$m,$n,$k,$size,$A->dtype(),$pDataA,$incA,$pDataY,$incY,$startAxis0,$sizeAxis0,$startAxis1,$sizeAxis1,$startAxis2,$sizeAxis2);
                break;
            }
        }
    }

    /**
    *  B(n,repeats,k) := A(n,k)
    */
    public function repeat(
        int $m,
        int $k,
        int $repeats,
        Buffer $A, int $offsetA,
        Buffer $B, int $offsetB
        ) : void
    {
        $this->assert_shape_parameter("m", $m);
        $this->assert_shape_parameter("k", $k);
        if($repeats<=0) {
            throw new InvalidArgumentException("Argument repeats must be greater than or equal 0.");
        }
    
        // Check Buffer A
        if($offsetA+$m*$k > $A->count()) {
            throw new InvalidArgumentException("Matrix A specification too large for buffer.");
        }
    
        // Check Buffer B
        if($offsetB+$m*$repeats*$k > $B->count()) {
            throw new InvalidArgumentException("Matrix B specification too large for buffer.");
        }
    
        // Check Buffer A and Y
        if($A->dtype()!=$B->dtype()) {
            $types = $this->dtypeToString[$A->dtype()].','.$this->dtypeToString[$B->dtype()];
            throw new InvalidArgumentException("Unmatch data type for A and B: ".$types);
        }
    
        switch ($A->dtype()) {
            case NDArray::float32: {
                $pDataA = $A->addr($offsetA);
                $pDataB = $B->addr($offsetB);
                $this->ffi->rindow_matlib_s_repeat($m,$k,$repeats,$pDataA,$pDataB);
                break;
            }
            case NDArray::float64: {
                $pDataA = $A->addr($offsetA);
                $pDataB = $B->addr($offsetB);
                $this->ffi->rindow_matlib_d_repeat($m,$k,$repeats,$pDataA,$pDataB);
                break;
            }
            default:{
                if(!$this->is_integer_dtype($A->dtype())&&
                    NDArray::bool!=$A->dtype()) {
                    throw new InvalidArgumentException("Unsupported data type.");
                }
                $pDataA = $A->addr($offsetA);
                $pDataB = $B->addr($offsetB);
                $this->ffi->rindow_matlib_i_repeat($m,$k,$repeats,$A->dtype(),$pDataA,$pDataB);
                break;
            }
        }
    }

    /**
    *   X(m) := sum( A(m,n) )
    */
    public function reduceSum(
        int $m,
        int $n,
        int $k,
        Buffer $A, int $offsetA,
        Buffer $B, int $offsetB
        ) : void
    {
        $this->assert_shape_parameter("m", $m);
        $this->assert_shape_parameter("n", $n);
        $this->assert_shape_parameter("k", $k);
        // Check Buffer A
        if($offsetA<0) {
            throw new InvalidArgumentException("Argument offsetA must be greater than or equals 0.");
        }
        if($offsetA+$m*$n*$k>$A->count()) {
            throw new InvalidArgumentException("Matrix specification too large for bufferA.");
        }
    
        // Check Buffer B
        if($offsetB<0) {
            throw new InvalidArgumentException("Argument offsetB must be greater than or equals 0.");
        }
        if($offsetB+$m*$k>$B->count()) {
            throw new InvalidArgumentException("Matrix specification too large for bufferB.");
        }
    
        // Check Buffer A and B
        if($A->dtype()!=$B->dtype()) {
            $types = $this->dtypeToString[$A->dtype()].','.$this->dtypeToString[$B->dtype()];
            throw new InvalidArgumentException("Unmatch data type for A and B: ".$types);
        }
    
        switch ($A->dtype()) {
            case NDArray::float32: {
                $pDataA = $A->addr($offsetA);
                $pDataB = $B->addr($offsetB);
                $this->ffi->rindow_matlib_s_reducesum($m,$n,$k,$pDataA,$pDataB);
                break;
            }
            case NDArray::float64: {
                $pDataA = $A->addr($offsetA);
                $pDataB = $B->addr($offsetB);
                $this->ffi->rindow_matlib_s_reducesum($m,$n,$k,$pDataA,$pDataB);
                break;
            }
            default: {
                throw new InvalidArgumentException("Unsupported data type.");
            }
        }
    }

    /**
     * X(m) := max( A(m,n) )
     */
    public function reduceMax(
        int $m,
        int $n,
        int $k,
        Buffer $A, int $offsetA,
        Buffer $B, int $offsetB
        ) : void
    {
        $this->assert_shape_parameter("m", $m);
        $this->assert_shape_parameter("n", $n);
        $this->assert_shape_parameter("k", $k);
        // Check Buffer A
        if($offsetA<0) {
            throw new InvalidArgumentException("Argument offsetA must be greater than or equal 0.");
        }
        if($offsetA+$m*$n*$k>$A->count()) {
            throw new InvalidArgumentException("Matrix specification too large for bufferA.");
        }
    
        // Check Buffer B
        if($offsetB<0) {
            throw new InvalidArgumentException("Argument offsetB must be greater than or equal 0.");
        }
        if($offsetB+$m*$k>$B->count()) {
            throw new InvalidArgumentException("Matrix specification too large for bufferB.");
        }
    
        // Check Buffer A and B
        if($A->dtype()!=$B->dtype()) {
            $types = $this->dtypeToString[$A->dtype()].','.$this->dtypeToString[$B->dtype()];
            throw new InvalidArgumentException("Unmatch data type for A and B: ".$types);
        }
    
        switch ($A->dtype()) {
            case NDArray::float32: {
                $pDataA = $A->addr($offsetA);
                $pDataB = $B->addr($offsetB);
                $this->ffi->rindow_matlib_s_reducemax($m,$n,$k,$pDataA,$pDataB);
                break;
            }
            case NDArray::float64: {
                $pDataA = $A->addr($offsetA);
                $pDataB = $B->addr($offsetB);
                $this->ffi->rindow_matlib_d_reducemax($m,$n,$k,$pDataA,$pDataB);
                break;
            }
            default: {
                throw new InvalidArgumentException("Unsupported data type.");
            }
        }
    }

    /**
     * X(m) := max( A(m,n) )
     */
    public function reduceArgMax(
        int $m,
        int $n,
        int $k,
        Buffer $A, int $offsetA,
        Buffer $B, int $offsetB
        ) : void
    {
        $this->assert_shape_parameter("m", $m);
        $this->assert_shape_parameter("n", $n);
        $this->assert_shape_parameter("k", $k);
        // Check Buffer A
        if($offsetA<0) {
            throw new InvalidArgumentException("Argument offsetA must be greater than or equal 0.");
        }
        if($offsetA+$m*$n*$k>$A->count()) {
            throw new InvalidArgumentException("Matrix specification too large for bufferA.");
        }

        // Check Buffer B
        if($offsetB<0) {
            throw new InvalidArgumentException("Argument offsetB must be greater than or equal 0.");
        }
        if($offsetB+$m*$k>$B->count()) {
            throw new InvalidArgumentException("Matrix specification too large for bufferB.");
        }
    
        switch ($A->dtype()) {
            case NDArray::float32: {
                $pDataA = $A->addr($offsetA);
                $pDataB = $B->addr($offsetB);
                $this->ffi->rindow_matlib_s_reduceargmax($m,$n,$k,$pDataA,$B->dtype(),$pDataB);
                break;
            }
            case NDArray::float64: {
                $pDataA = $A->addr($offsetA);
                $pDataB = $B->addr($offsetB);
                $this->ffi->rindow_matlib_d_reduceargmax($m,$n,$k,$pDataA,$B->dtype(),$pDataB);
                break;
            }
            default: {
                throw new InvalidArgumentException("Unsupported data type.");
            }
        }
    }

    /**
    */
    public function randomUniform(
        int $n,
        Buffer $X, int $offsetX, int $incX,
        float|int $low,
        float|int $high,
        int $seed
        )
    {
        // Check Buffer X
        $this->assert_vector_buffer_spec("X", $X,$n,$offsetX,$incX);
    
        if($this->is_float_dtype($X->dtype())) {
            $low = (float)$low;
            $high = (float)$high;
        } else if($this->is_integer_dtype($X->dtype())) {
            $low = (int)$low;
            $high = (int)$high;
        } else {
            throw new InvalidArgumentException("Unsupported data type.");
        }
    
        switch($X->dtype()) {
            case NDArray::float32: {
                $pDataX = $X->addr($offsetX);
                $this->ffi->rindow_matlib_s_randomuniform($n,$pDataX,$incX,$low,$high,$seed);
                break;
            }
            case NDArray::float64: {
                $pDataX = $X->addr($offsetX);
                $this->ffi->rindow_matlib_d_randomuniform($n,$pDataX,$incX,$low,$high,$seed);
                break;
            }
            default: {
                $pDataX = $X->addr($offsetX);
                $this->ffi->rindow_matlib_i_randomuniform($n,$X->dtype(),$pDataX,$incX,$low,$high,$seed);
                break;
            }
        }
    }

    /**
    */
    public function randomNormal(
        int $n,
        Buffer $X, int $offsetX, int $incX,
        float $mean,
        float $scale,
        int $seed
        )
    {
        // Check Buffer X
        $this->assert_vector_buffer_spec("X", $X,$n,$offsetX,$incX);
    
        switch($X->dtype()) {
            case NDArray::float32: {
                $pDataX = $X->addr($offsetX);
                $this->ffi->rindow_matlib_s_randomnormal($n,$pDataX,$incX,$mean,$scale,$seed);
                break;
            }
            case NDArray::float64: {
                $pDataX = $X->addr($offsetX);
                $this->ffi->rindow_matlib_d_randomnormal($n,$pDataX,$incX,$mean,$scale,$seed);
                break;
            }
            default: {
                throw new InvalidArgumentException("Unsupported data type.");
            }
        }
    }

    /**
    */
    public function randomSequence(
        int $n,
        int $size,
        Buffer $X, int $offsetX, int $incX,
        int $seed
        )
    {
        // Check Buffer X
        $this->assert_vector_buffer_spec("X", $X,$n,$offsetX,$incX);
        if($n<$size||$size<1) {
            throw new InvalidArgumentException("size must be smaller then n or equal.");
        }
        if(!$this->is_integer_dtype($X->dtype())) {
            throw new InvalidArgumentException("dtype must be integer dtype.");
        }
    
        $pDataX = $X->addr($offsetX);
        $this->ffi->rindow_matlib_i_randomsequence($n,$size,$X->dtype(),$pDataX,$incX,$seed);
    }

    /**
    * images: (n,h,w,c) : channels_last
    *        (n,c,h,w) : channels_first
    * strides:
    * padding:
    * data_format:
    * output:(n,i)
    */
    public function im2col1d(
        bool $reverse,
        Buffer $images,
        int $images_offset,
        int $images_size,
        int $batches,
        int $im_w,
        int $channels,
        int $filter_w,
        int $stride_w,
        bool $padding,
        bool $channels_first,
        int $dilation_w,
        bool $cols_channels_first,
        Buffer $cols,
        int $cols_offset,
        int $cols_size
        )
    {
        $this->assert_buffer_size($images, $images_offset, $images_size,
            "Invalid images buffer offset or size");
        $this->assert_buffer_size($cols, $cols_offset, $cols_size,
            "Invalid cols buffer offset or size");
        if($images->dtype()!=NDArray::float32 &&
            $images->dtype()!=NDArray::float64) {
            throw new InvalidArgumentException("Unsupported data type");
        }
        // Check dtype and Buffer Y
        if($images->dtype()!=$cols->dtype()) {
            $types = $this->dtypeToString[$images->dtype()].','.$this->dtypeToString[$cols->dtype()];
            throw new InvalidArgumentException("Unmatch data type of images and cols: ".$types);
        }
        if($images->count()<$images_offset+$images_size) {
            throw new InvalidArgumentException("Images size is out of range");
        }
    
        $pDataImages = $images->addr($images_offset);
        $pDataCols = $cols->addr($cols_offset);
    
        $rc = $this->ffi->rindow_matlib_im2col1d(
            $images->dtype(),
            $reverse,
            $pDataImages,
            $images_size,
            $batches,
            $im_w,
            $channels,
            $filter_w,
            $stride_w,
            $padding,
            $channels_first,
            $dilation_w,
            $cols_channels_first,
            $pDataCols,
            $cols_size
        );
        switch($rc) {
            case 0: {
                break;
            }
            case self::E_UNSUPPORTED_DATA_TYPE: {
                throw new InvalidArgumentException("Unsupported data type.");
            }
            case self::E_INVALID_SHAPE_OR_PARAM: {
                throw new InvalidArgumentException("Invalid shape or parameters.");
            }
            case self::E_UNMATCH_COLS_BUFFER_SIZE: {
                throw new InvalidArgumentException("Unmatch cols buffer size and images shape");
            }
            case self::E_UNMATCH_IMAGE_BUFFER_SIZE: {
                throw new InvalidArgumentException("Unmatch images buffer size and images shape");
            }
            case self::E_IMAGES_OUT_OF_RANGE: {
                throw new RuntimeException("Images data out of range");
            }
            case self::E_COLS_OUT_OF_RANGE: {
                throw new RuntimeException("Cols data out of range");
            }
            default: {
                throw new RuntimeException(sprintf("Unkown Error (%d)", $rc));
            }
        }
    }

    /**
    * images: (n,h,w,c) : channels_last
    *        (n,c,h,w) : channels_first
    * strides:
    * padding:
    * data_format:
    * output:(n,i)
    */
    public function im2col2d(
        bool $reverse,
        Buffer $images,
        int $images_offset,
        int $images_size,
        int $batches,

        int $im_h,
        int $im_w,
        int $channels,
        int $filter_h,
        int $filter_w,

        int $stride_h,
        int $stride_w,
        bool $padding,
        bool $channels_first,
        int $dilation_h,

        int $dilation_w,
        bool $cols_channels_first,
        Buffer $cols,
        int $cols_offset,
        int $cols_size
        )
    {
        $this->assert_buffer_size($images, $images_offset, $images_size,
            "Invalid images buffer offset or size");
        $this->assert_buffer_size($cols, $cols_offset, $cols_size,
            "Invalid cols buffer offset or size");
        if($images->dtype()!=NDArray::float32 &&
            $images->dtype()!=NDArray::float64) {
            throw new InvalidArgumentException("Unsupported data type");
        }

        // Check dtype and Buffer Y
        if($images->dtype()!=$cols->dtype()) {
            $types = $this->dtypeToString[$images->dtype()].','.$this->dtypeToString[$cols->dtype()];
            throw new InvalidArgumentException("Unmatch data type of images and cols: ".$types);
        }
        if($images->count()<$images_offset+$images_size) {
            throw new InvalidArgumentException("Images size is out of range");
        }

        $pDataImages = $images->addr($images_offset);
        $pDataCols = $cols->addr($cols_offset);
    
        $rc = $this->ffi->rindow_matlib_im2col2d(
            $images->dtype(),
            $reverse,
            $pDataImages,
            $images_size,
            $batches,
    
            $im_h,
            $im_w,
            $channels,
            $filter_h,
            $filter_w,
    
            $stride_h,
            $stride_w,
            $padding,
            $channels_first,
            $dilation_h,
    
            $dilation_w,
            $cols_channels_first,
            $pDataCols,
            $cols_size
        );
        switch($rc) {
            case 0: {
                break;
            }
            case self::E_UNSUPPORTED_DATA_TYPE: {
                throw new InvalidArgumentException("Unsupported data type.");
            }
            case self::E_INVALID_SHAPE_OR_PARAM: {
                throw new InvalidArgumentException("Invalid shape or parameters.");
            }
            case self::E_UNMATCH_COLS_BUFFER_SIZE: {
                throw new InvalidArgumentException("Unmatch cols buffer size and images shape");
            }
            case self::E_UNMATCH_IMAGE_BUFFER_SIZE: {
                throw new InvalidArgumentException("Unmatch images buffer size and images shape");
            }
            case self::E_IMAGES_OUT_OF_RANGE: {
                throw new RuntimeException("Images data out of range");
            }
            case self::E_COLS_OUT_OF_RANGE: {
                throw new RuntimeException("Cols data out of range");
            }
            default: {
                throw new RuntimeException(sprintf("Unkown Error (%d)", $rc));
            }
        }
    }

    /**
    * images: (n,h,w,c) : channels_last
    *        (n,c,h,w) : channels_first
    * strides:
    * padding:
    * data_format:
    * output:(n,i)
    */
    public function im2col3d(
        bool $reverse,
        Buffer $images,
        int $images_offset,
        int $images_size,
        int $batches,
        int $im_d,
        int $im_h,
        int $im_w,
        int $channels,
        int $filter_d,
        int $filter_h,
        int $filter_w,
        int $stride_d,
        int $stride_h,
        int $stride_w,
        bool $padding,
        bool $channels_first,
        int $dilation_d,
        int $dilation_h,
        int $dilation_w,
        bool $cols_channels_first,
        Buffer $cols,
        int $cols_offset,
        int $cols_size
        )
    {
        $this->assert_buffer_size($images, $images_offset, $images_size,
            "Invalid images buffer offset or size");
        $this->assert_buffer_size($cols, $cols_offset, $cols_size,
            "Invalid cols buffer offset or size");
        if($images->dtype()!=NDArray::float32 &&
            $images->dtype()!=NDArray::float64) {
            throw new InvalidArgumentException("Unsupported data type");
        }

        // Check dtype and Buffer Y
        if($images->dtype()!=$cols->dtype()) {
            $types = $this->dtypeToString[$images->dtype()].','.$this->dtypeToString[$cols->dtype()];
            throw new InvalidArgumentException("Unmatch data type of images and cols: ".$types);
        }
        if($images->count()<$images_offset+$images_size) {
            throw new InvalidArgumentException("Images size is out of range");
        }

        $pDataImages = $images->addr($images_offset);
        $pDataCols = $cols->addr($cols_offset);

        $rc = $this->ffi->rindow_matlib_im2col3d(
            $images->dtype(),
            $reverse,
            $pDataImages,
            $images_size,
            $batches,
    
            $im_d,
            $im_h,
            $im_w,
            $channels,
            $filter_d,
    
            $filter_h,
            $filter_w,
            $stride_d,
            $stride_h,
            $stride_w,
    
            $padding,
            $channels_first,
            $dilation_d,
            $dilation_h,
            $dilation_w,
    
            $cols_channels_first,
            $pDataCols,
            $cols_size
        );
    
        switch($rc) {
            case 0: {
                break;
            }
            case self::E_UNSUPPORTED_DATA_TYPE: {
                throw new InvalidArgumentException("Unsupported data type.");
            }
            case self::E_INVALID_SHAPE_OR_PARAM: {
                throw new InvalidArgumentException("Invalid shape or parameters.");
            }
            case self::E_UNMATCH_COLS_BUFFER_SIZE: {
                throw new InvalidArgumentException("Unmatch cols buffer size and images shape");
            }
            case self::E_UNMATCH_IMAGE_BUFFER_SIZE: {
                throw new InvalidArgumentException("Unmatch images buffer size and images shape");
            }
            case self::E_IMAGES_OUT_OF_RANGE: {
                throw new RuntimeException("Images data out of range");
            }
            case self::E_COLS_OUT_OF_RANGE: {
                throw new RuntimeException("Cols data out of range");
            }
            default: {
                throw new RuntimeException(sprintf("Unkown Error (%d)", $rc));
            }
        }
    }
}
