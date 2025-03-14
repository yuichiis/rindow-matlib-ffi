<?php
namespace RindowTest\Matlib\FFI\MatlibTest;

use PHPUnit\Framework\TestCase;
use PHPUnit\Framework\Attributes\DataProvider;
use Interop\Polite\Math\Matrix\NDArray;
use Interop\Polite\Math\Matrix\BLAS;
use Rindow\Math\Buffer\FFI\Buffer;
use Rindow\Matlib\FFI\MatlibFactory;
use Rindow\Matlib\FFI\Matlib;
use InvalidArgumentException;
use RuntimeException;
use LogicException;
use OutOfRangeException;
use TypeError;
use ArrayObject;
use ArrayAccess;
use SplFixedArray;

function R(
    int $start,
    int $limit,
) : Range
{
    if(func_num_args()!=2) {
        throw new InvalidArgumentException('R must have only two arguments: "start" and "limit".');
    }
    return new Range(start:$start,limit:$limit);
}

class Range
{
    protected mixed $start;
    protected mixed $limit;
    protected mixed $delta;

    public function __construct(
        int|float $limit,
        int|float|null $start=null,
        int|float|null $delta=null)
    {
        $this->limit = $limit;
        $this->start = $start ?? 0;
        $this->delta = $delta ?? (($limit>=$start)? 1 : -1);
    }

    public function start() : mixed
    {
        return $this->start;
    }

    public function limit() : mixed
    {
        return $this->limit;
    }

    public function delta() : mixed
    {
        return $this->delta;
    }
}

class MatlibTest extends TestCase
{
    public function getMatlib()
    {
        $factory = new MatlibFactory();
        $matlib = $factory->Matlib();
        return $matlib;
    }

    public function zeros(array $shape,?int $dtype=null)
    {
        $ndarray = $this->array(null,$dtype,$shape);
        $buffer = $ndarray->buffer();
        $size = $buffer->count();
        for($i=0;$i<$size;$i++) {
            $buffer[$i] = 0;
        }
        return $ndarray;
    }

    public function ones(array $shape,?int $dtype=null)
    {
        $ndarray = $this->array(null,$dtype,$shape);
        $buffer = $ndarray->buffer();
        $size = $buffer->count();
        if($dtype==NDArray::bool) {
            $value = true;
        } else {
            $value = 1;
        }
        for($i=0;$i<$size;$i++) {
            $buffer[$i] = $value;
        }
        return $ndarray;
    }

    public function zerosLike(object $ndarray)
    {
        return $this->zeros($ndarray->shape(),$ndarray->dtype());
    }

    public function arange(int $count ,$start=null, $step=null, $dtype=null)
    {
        if($start===null)
            $start = 0;
        if($step===null)
            $step = 1;
        if($dtype===null) {
            if(is_int($start))
                $dtype = NDArray::int32;
            else
                $dtype = NDArray::float32;
        }
        $array = $this->zeros([$count], $dtype);
        $buffer = $array->buffer();
        $n = $start;
        for($i=0; $i<$count; $i++) {
            $buffer[$i] = $n;
            $n += $step;
        }
        return $array;
    }

    public function array(int|float|array|null $array=null, ?int $dtype=null, ?array $shape=null) : object
    {
        $ndarray = new class ($array, $dtype, $shape) implements NDArray {
            protected object $buffer;
            protected int $size;
            protected int $dtype;
            protected int $offset;
            protected array $shape;
            public function __construct(int|float|array|Buffer|null $array=null, ?int $dtype=null, ?array $shape=null, ?int $offset=null) {
                $dtype = $dtype ?? NDArray::float32;
                $offset = $offset ?? 0;
                if(is_array($array)||$array instanceof ArrayObject) {
                    $dummyBuffer = new ArrayObject();
                    $idx = 0;
                    $this->array2Flat($array,$dummyBuffer,$idx,$prepare=true);
                    $buffer = $this->newBuffer($idx,$dtype);
                    $idx = 0;
                    $this->array2Flat($array,$buffer,$idx,$prepare=false);
                    if($shape===null) {
                        $shape = $this->genShape($array);
                    }
                } elseif(is_numeric($array)||is_bool($array)) {
                    if(is_bool($array)&&$dtype!=NDArray::bool) {
                        throw new InvalidArgumentException("unmatch dtype with bool value");
                    }
                    $buffer = $this->newBuffer(1,$dtype);
                    $buffer[0] = $array;
                    if($shape===null) {
                        $shape = [];
                    }
                    $this->checkShape($shape);
                    if(array_product($shape)!=1)
                        throw new InvalidArgumentException("Invalid dimension size");
                } elseif($array===null && $shape!==null) {
                    $this->checkShape($shape);
                    $size = (int)array_product($shape);
                    $buffer = $this->newBuffer($size,$dtype);
                } elseif($array===null && $shape!==null) {
                    $this->checkShape($shape);
                    $size = (int)array_product($shape);
                    $buffer = $this->newBuffer($size,$dtype);
                } elseif($this->isBuffer($array)) {
                    if($offset===null||!is_int($offset))
                        throw new InvalidArgumentException("Must specify offset with the buffer");
                    if($shape===null)
                        throw new InvalidArgumentException("Invalid dimension size");
                    $buffer = $array;
                } else {
                    var_dump($array);var_dump($shape);
                    throw new \Exception("Illegal array type");
                }
                $this->buffer = $buffer;
                $this->size = $buffer->count();
                $this->dtype = $buffer->dtype();
                $this->shape = $shape;
                $this->offset = $offset;
            }

            function newBuffer($size,$dtype) : object
            {
                return new Buffer($size,$dtype);
            }
            
            protected function isBuffer($buffer)
            {
                if($buffer instanceof SplFixedArray || $buffer instanceof Buffer) {
                    return true;
                } else {
                    return false;
                }
            }
        
            protected function array2Flat($A, $F, &$idx, $prepare)
            {
                if(is_array($A)) {
                    ksort($A);
                } elseif($A instanceof ArrayObject) {
                    $A->ksort();
                }
        
                $num = null;
                foreach ($A as $key => $value) {
                    if(!is_int($key))
                        throw new InvalidArgumentException("Dimension must be integer");
                    if(is_array($value)||$value instanceof ArrayObject) {
                        $num2 = $this->array2Flat($value, $F, $idx, $prepare);
                        if($num===null) {
                            $num = $num2;
                        } else {
                            if($num!=$num2)
                                throw new InvalidArgumentException("The shape of the dimension is broken");
                        }
                    } else {
                        if($num!==null)
                            throw new InvalidArgumentException("The shape of the dimension is broken");
                        if(!$prepare)
                            $F[$idx] = $value;
                        $idx++;
                    }
                }
                return count($A);
            }

            protected function flat2Array($F, &$idx, array $shape)
            {
                $size = array_shift($shape);
                if(count($shape)) {
                    $A = [];
                    for($i=0; $i<$size; $i++) {
                        $A[$i] = $this->flat2Array($F,$idx,$shape);
                    }
                }  else {
                    $A = [];
                    for($i=0; $i<$size; $i++) {
                        $A[$i] = $F[$idx];
                        $idx++;
                    }
                }
                return $A;
            }
                
            protected function genShape($A)
            {
                $shape = [];
                while(is_array($A) || $A instanceof ArrayObject) {
                    $shape[] = count($A);
                    $A = $A[0];
                }
                return $shape;
            }
        
            protected function checkShape(array $shape)
            {
                foreach($shape as $num) {
                    if(!is_int($num)) {
                        throw new InvalidArgumentException(
                            "Invalid shape numbers. It gives ".gettype($num));
                    }
                    if($num<=0) {
                        throw new InvalidArgumentException(
                            "Invalid shape numbers. It gives ".$num);
                    }
                }
            }

            public function toArray()
            {
                if(count($this->shape)==0) {
                    return $this->buffer[$this->offset];
                }
                $idx = $this->offset;
                return $this->flat2Array($this->buffer, $idx, $this->shape);
            }

            public function shape() : array { return $this->shape; }

            public function ndim() : int { return count($this->shape); }
        
            public function dtype() { return $this->dtype; }
        
            public function buffer() : ArrayAccess { return $this->buffer; }
        
            public function offset() : int { return $this->offset; }
        
            public function size() : int { return $this->buffer->count(); }
        
            public function reshape(array $shape) : NDArray
            {
                if(array_product($shape)==array_product($this->shape)) {
                    $this->shape = $shape;
                } else {
                    throw new \Exception("unmatch shape");
                }
                return $this;
            }
            public function offsetExists( $offset ) : bool { throw new \Excpetion('not implement'); }
            public function offsetGet( $offset ) : mixed
            {
                if(is_array($offset)) {
                    throw new InvalidArgumentException("offset style is old renge style.");
                }
                // for single index specification
                if(is_numeric($offset)) {
                    $shape = $this->shape;
                    $max = array_shift($shape);
                    if(count($shape)==0) {
                        return $this->buffer[$this->offset+$offset];
                    }
                    $size = (int)array_product($shape);
                    $new = new self($this->buffer,$this->dtype,$shape,$this->offset+$offset*$size);
                    return $new;
                }

                // for range spesification
                $shape = $this->shape;
                array_shift($shape);
                if(is_array($offset)) {
                    $start = $offset[0];
                    $limit = $offset[1]+1;
                } else {
                    $start = $offset->start();
                    $limit = $offset->limit();
                }
                $rowsCount = $limit-$start;
                if(count($shape)>0) {
                    $itemSize = (int)array_product($shape);
                } else {
                    $itemSize = 1;
                }
                if($rowsCount<0) {
                    throw new OutOfRangeException('Invalid range');
                }
                array_unshift($shape,$rowsCount);
                $size = (int)array_product($shape);
                $new = new self($this->buffer,$this->dtype,$shape,$this->offset+$start*$itemSize);
                return $new;
            }
            public function offsetSet( $offset , $value ) : void { throw new \Exception('not implement'); }
            public function offsetUnset( $offset ) : void { throw new \Exception('not implement'); }
            public function count() : int
            {
                return $this->buffer->count();
            }
            public function  getIterator() : Traversable  { throw new \Exception('not implement'); }
        };
        return $ndarray;
    }


    public function getMatlibVersion($matlib)
    {
        $config = $matlib->getConfig();
        if(strpos($config,'Matlib')===0) {
            $config = explode(' ',$config);
            return $config[1];
        } else {
            return '0.0.0';
        }
    }

    public function checkSkip($mark)
    {
        if(!in_array($mark,[
            //'multiply',
            //'duplicate'
            ])) {
            return false;
        }
        $this->markTestSkipped($mark);
        return true;
    }


    public function sum($n,$X,$offsetX,$incX)
    {
        $a = 0;
        for($i=0;$i<$n;$i++) {
            $a += $X[$offsetX+$i*$incX];
        }
        return $a;
    }

    protected function printableShapes($values)
    {
        if(!is_array($values)) {
            if($values instanceof NDArray)
                return '('.implode(',',$values->shape()).')';
            if(is_object($values))
                return '"'.get_class($values).'"';
            if(is_numeric($values) || is_string($values))
                return strval($values);
            return gettype($values);
        }
        $string = '[';
        foreach($values as $value) {
            if($string!='[') {
                $string .= ',';
            }
            $string .= $this->printableShapes($value);
        }
        $string .= ']';
        return $string;
    }

    public function translate_amin(
        NDArray $X) : array
    {
        $N = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();
        return [$N,$XX,$offX,1];
    }

    public function translate_increment(
        NDArray $X,
        ?float $beta=null,
        ?float $alpha=null) : array
    {
        $N = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();
        if($alpha===null) {
            $alpha = 1.0;
        }
        if($beta===null) {
            $beta = 0.0;
        }
        return [$N,$alpha,$XX,$offX,1,$beta];
    }

    public function translate_maximum(
        NDArray $A,
        NDArray $X,
        ) : array
    {
        [$m,$n] = $A->shape();
        $AA = $A->buffer();
        $offA = $A->offset();
        $XX = $X->buffer();
        $offX = $X->offset();

        return [$m,$n,$AA,$offA,$n,$XX,$offX,1];
    }

    public function translate_nan2num(
        NDArray $X,
        float $alpha
        ) : array
    {
        $n = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();

        return [
            $n,
            $XX,$offX,1,
            $alpha
        ];
    }

    public function translate_multiply(
       NDArray $X,
       NDArray $A,
       ?bool $trans=null
       ) : array
    {
        if($trans===null)
            $trans = false;
        $shapeX = $X->shape();
        $shapeA = $A->shape();
        if($trans)
            $shapeA = array_reverse($shapeA);
        while(true) {
            $xd = array_pop($shapeX);
            if($xd===null)
                break;
            $ad = array_pop($shapeA);
            if($xd!==$ad)
                throw new InvalidArgumentException('Unmatch dimension size for broadcast.: '.
                    '['.implode(',',$X->shape()).'] ['.implode(',',$A->shape()).']');
        }
        $n = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();
        $m = $A->size()/$n;
        $AA = $A->buffer();
        $offA = $A->offset();
        if($trans) {
            [$m,$n] = [$n,$m];
        }

        return [
            $trans,
            $m,
            $n,
            $XX,$offX,1,
            $AA,$offA,$n,
        ];
    }

    public function translate_add(
       NDArray $X,
       NDArray $A,
       ?float $alpha=null,
       ?bool $trans=null
       ) : array
    {
        if($trans===null)
            $trans = false;
        if($alpha===null)
            $alpha = 1.0;
        $shapeX = $X->shape();
        $shapeA = $A->shape();
        if($trans)
            $shapeA = array_reverse($shapeA);
        while(true) {
            $xd = array_pop($shapeX);
            if($xd===null)
                break;
            $ad = array_pop($shapeA);
            if($xd!==$ad)
                throw new InvalidArgumentException('Unmatch dimension size for broadcast.: '.
                    '['.implode(',',$X->shape()).'] ['.implode(',',$A->shape()).']');
        }
        $n = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();
        $m = $A->size()/$n;
        $AA = $A->buffer();
        $offA = $A->offset();
        if($trans) {
            [$m,$n] = [$n,$m];
        }

        return [
            $trans,
            $m,
            $n,
            $alpha,
            $XX,$offX,1,
            $AA,$offA,$n,
        ];
    }

    public function translate_duplicate(
        NDArray $X, ?int $n=null, ?bool $trans=null, ?NDArray $A=null) : array
    {
        if($trans===null)
            $trans = false;
        if($A===null) {
            if(!$trans) {
                $A = $this->alloc(array_merge([$n],$X->shape()));
            } else {
                $A = $this->alloc(array_merge($X->shape(),[$n]));
            }
        } else {
            $shapeX = $X->shape();
            $shapeA = $A->shape();
            if($trans)
                $shapeA = array_reverse($shapeA);
            while(true) {
                $xd = array_pop($shapeX);
                if($xd===null)
                    break;
                $ad = array_pop($shapeA);
                if($xd!==$ad)
                    throw new InvalidArgumentException('Unmatch dimension size for broadcast.: '.
                        '['.implode(',',$X->shape()).'] => ['.implode(',',$A->shape()).']');
            }
        }

        $n = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();
        $m = $A->size()/$n;
        $AA = $A->buffer();
        $offA = $A->offset();
        if($trans) {
            [$m,$n] = [$n,$m];
        }

        return [
            $trans,
            $m,
            $n,
            $XX,$offX,1,
            $AA,$offA,$n
        ];
    }

    public function translate_matrixcopy(
        NDArray $A,
        ?bool $trans=null,
        ?float $alpha=null,
        ?NDArray $B=null
         ) : array
    {
        $trans = $trans ?? false;
        $alpha = $alpha ?? 1.0;

        $shape = $A->shape();
        if($trans) {
            $shape = [$shape[1],$shape[0]];
        }
        if($B==null) {
            $B = $this->zeros($shape,$A->dtype());
        } else {
            if($B->shape()!=$shape) {
                throw new InvalidArgumentException('output shape must be transpose matrix of input.');
            }
            if($B->dtype()!=$A->dtype()) {
                throw new InvalidArgumentException('output data type must be same with matrix of input.');
            }
        }
        $m = $shape[1];
        $n = $shape[0];
        $AA = $A->buffer();
        $offA = $A->offset();
        $ldA = $n;
        $BB = $B->buffer();
        $offB = $B->offset();
        $ldB = $trans ? $m : $n;
        return [
            $trans,
            $m,
            $n,
            $alpha,
            $AA,$offA,$ldA,
            $BB,$offB,$ldB
        ];
    }

    public function translate_imagecopy(
        NDArray $A,
        ?NDArray $B=null,
        ?bool $channels_first=null,
        ?int $heightShift=null,
        ?int $widthShift=null,
        ?bool $verticalFlip=null,
        ?bool $horizontalFlip=null,
        ?bool $rgbFlip=null
        ) : array
    {
        if($A->ndim()!=3) {
            throw new InvalidArgumentException('input array must be 3D.');
        }
        $shape = $A->shape();
        if($B==null) {
            $B = $this->alloc($shape,$A->dtype());
            $this->zeros($B);
        } else {
            if($B->shape()!=$shape) {
                throw new InvalidArgumentException('output shape must be transpose matrix of input.');
            }
            if($B->dtype()!=$A->dtype()) {
                throw new InvalidArgumentException('output data type must be same with matrix of input.');
            }
        }
        if($heightShift==null) {
            $heightShift=0;
        }
        if($widthShift==null) {
            $widthShift=0;
        }
        if($verticalFlip==null) {
            $verticalFlip=false;
        }
        if($horizontalFlip==null) {
            $horizontalFlip=false;
        }
        if($rgbFlip==null) {
            $rgbFlip=false;
        }
        if($channels_first==null) {
            $channels_first=false;
            $height = $shape[0];
            $width = $shape[1];
            $channels = $shape[2];
        } else {
            $channels_first=true;
            $channels = $shape[0];
            $height = $shape[1];
            $width = $shape[2];
        }
        $AA = $A->buffer();
        $offA = $A->offset();
        $BB = $B->buffer();
        $offB = $B->offset();
        return [
            $height,
            $width,
            $channels,
            $AA, $offA,
            $BB, $offB,
            $channels_first,
            $heightShift,
            $widthShift,
            $verticalFlip,
            $horizontalFlip,
            $rgbFlip
        ];
    }

    public function translate_pow(
       NDArray $A,
       NDArray $alpha,
       ?bool $trans=null
       ) : array
    {
        if($trans===null) {
            $trans = false;
        }
        $shapeA = $A->shape();
        if(is_numeric($alpha)) {
            $alpha = $this->array($alpha,dtype:$A->dtype());
        }
        $shapeX = $alpha->shape();
        if(count($shapeX)==0) {
            $trans = false;
            $shapeA = [(int)array_product($shapeA),1];
            $shapeX = [1];
        }

        if($trans) {
            $shapeA = array_reverse($shapeA);
        }
        while(true) {
            $xd = array_pop($shapeX);
            if($xd===null)
                break;
            $ad = array_pop($shapeA);
            if($xd!==$ad) {
                $shapeA = $trans ? array_reverse($A->shape()) : $A->shape();
                throw new InvalidArgumentException('Unmatch dimension size for broadcast.: '.
                    '['.implode(',',$X->shape()).'] => ['.implode(',',$shapeA).']');
            }
        }
        $n = $alpha->size();
        $XX = $alpha->buffer();
        $offX = $alpha->offset();
        $m = $A->size()/$n;
        $AA = $A->buffer();
        $offA = $A->offset();
        if($trans) {
            [$m,$n] = [$n,$m];
        }

        return [
            $trans,
            $m,
            $n,
            $AA,$offA,$n,
            $XX,$offX,1,
        ];
    }

    public function translate_square(
        NDArray $X
        ) : array
    {
        $n = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();

        return [
            $n,
            $XX,$offX,1
        ];
    }

    public function translate_fill(
        $value,
        NDArray $X
        )
    {
        if(is_scalar($value)) {
            if(is_string($value)) {
                $value = ord($value);
            }
            $V = $this->alloc([1],$X->dtype());
            $V[0] = $value;
        } elseif($value instanceof NDArray) {
            if($value->size()!=1) {
                throw new InvalidArgumentException('Value must be scalar');
            }
            $V = $value;
        } else {
            throw new InvalidArgumentException('Invalid data type');
        }
        $n = $X->size();
        $VV = $V->buffer();
        $offV = $V->offset();
        $XX = $X->buffer();
        $offX = $X->offset();
        return [
            $n,
            $VV, $offV,
            $XX, $offX,1
        ];
    }

    /*
    public function translate_selectAxis0(
        NDArray $A,
        NDArray $X,
        NDArray $Y=null) : array
    {
        if($X->ndim()!=1) {
            throw new InvalidArgumentException('"X" must be 1D-NDArray.');
        }
        $countX = $X->shape()[0];
        if($A->ndim()==1) {
            $shape = $X->shape();
            $m = $A->shape()[0];
            $n = 1;
        } else {
            $shape = $A->shape();
            $m = $shape[0];
            $n = (int)($A->size()/$m);
            array_shift($shape);
            array_unshift($shape,$countX);
        }
        if($Y===null) {
            $Y = $this->alloc($shape,$A->dtype());
        } else {
            if($Y->shape()!=$shape) {
                throw new InvalidArgumentException('Unmatch size "Y" with "X" and "A" .');
            }
        }

        //if($A->ndim()==1) {
        //    $A = $A->reshape([$n,1]);
        //}
        //if($Y->ndim()==1) {
        //    $newY = $Y->reshape([$n,1]);
        //} else {
        //    $newY = $Y;
        //}
        //for($i=0;$i<$n;$i++) {
        //    $this->copy($A[$X[$i]],$newY[$i]);
        //}
        //return $Y;

        $AA = $A->buffer();
        $offA = $A->offset();
        $ldA = $n;
        $XX = $X->buffer();
        $offX = $X->offset();
        $YY = $Y->buffer();
        $offY = $Y->offset();
        $ldY = $n;

        return [
            $m,
            $n,
            $countX,
            $AA,$offA,$ldA,
            $XX,$offX,1,
            $YY,$offY,$ldY];
    }

    public function translate_selectAxis1(
        NDArray $A,
        NDArray $X,
        NDArray $Y=null) : array
    {
        if($A->ndim()!=2) {
            throw new InvalidArgumentException('"A" must be 2D-NDArray.');
        }
        if($X->ndim()!=1) {
            throw new InvalidArgumentException('"X" must be 1D-NDArray.');
        }
        [$m,$n] = $A->shape();
        if($X->size()!=$m) {
            throw new InvalidArgumentException('Unmatch size "X" with rows of "A".');
        }
        if($Y==null) {
            $Y = $this->alloc([$m],$A->dtype());
        }
        if($Y->ndim()!=1) {
            throw new InvalidArgumentException('"Y" must be 1D-NDArray.');
        }
        if($Y->size()!=$m) {
            throw new InvalidArgumentException('Unmatch size "Y" with rows of "A".');
        }

        $AA = $A->buffer();
        $offA = $A->offset();
        $ldA = $n;
        $XX = $X->buffer();
        $offX = $X->offset();
        $YY = $Y->buffer();
        $offY = $Y->offset();

        return [
            $m,
            $n,
            $AA,$offA,$ldA,
            $XX,$offX,1,
            $YY,$offY,1,
        ];
    }
*/
    public function translate_gather(
        bool $scatterAdd,
        NDArray $A,
        NDArray $X,
        ?int $axis=null,
        ?NDArray $B=null,
        ?int $dtype=null) : array
    {
//echo "shapeX=[".implode(',',$X->shape())."],shapeA=[".implode(',',$A->shape())."]\n";
        if($axis===null) {
            $postfixShape = $A->shape();
            $prefixShape = $X->shape();
            $numClass = array_shift($postfixShape);
            $m = 1;
            $n = array_product($prefixShape);
            $k = array_product($postfixShape);
            $reductionDims = false;
            $outputShape = array_merge($prefixShape,$postfixShape);
        } else {
            $ndim = $A->ndim();
            $orgAxis = $axis;
            if($axis<0) {
                $axis = $ndim+$axis;
            }
            $postfixShape = $A->shape();
            $prefixShape = [];
            for($i=0;$i<$axis;$i++) {
                $prefixShape[] = array_shift($postfixShape);
            }
            $numClass = array_shift($postfixShape);
            $m = array_product($prefixShape);
            $n = array_product($postfixShape);
            $k = 1;
            $reductionDims = true;
            $outputShape = array_merge($prefixShape,$postfixShape);
            if($X->shape()!=$outputShape) {
                throw new InvalidArgumentException('Unmatch Shape:'.
                                        $this->printableShapes([$A,$X]));
            }
        }
//echo "outputShape=[".implode(',',$outputShape)."]\n";
        if($dtype===null) {
            $dtype = $A->dtype();
        }
        if($B==null) {
            $B = $this->alloc($outputShape,$dtype);
            $this->zeros($B);
        } else {
            if($B->shape()!=$outputShape) {
                throw new InvalidArgumentException("Unmatch output shape of dimension: ".
                                            $this->printableShapes([$outputShape,$B]));
            }
        }

        $AA = $A->buffer();
        $offA = $A->offset();
        $XX = $X->buffer();
        $offX = $X->offset();
        $BB = $B->buffer();
        $offB = $B->offset();

        if($scatterAdd) {
            $reverse=true;
            $addMode=true;
        } else {
            $reverse=false;
            $addMode=false;
        }
        if($reductionDims) {
            return [ $reduce=true,
                $reverse,
                $addMode,
                $m,
                $n,
                $numClass,
                $XX,$offX,
                $AA,$offA,
                $BB,$offB];
        } else {
            return [ $reduce=false,
                $reverse,
                $addMode,
                $n,
                $k,
                $numClass,
                $XX,$offX,
                $AA,$offA,
                $BB,$offB];
        }
    }

    public function translate_scatter(
        NDArray $X,
        NDArray $A,
        int $numClass,
        ?int $axis=null,
        ?NDArray $B=null,
        ?int $dtype=null) : array
    {
//echo "shapeX=[".implode(',',$X->shape())."],shapeA=[".implode(',',$A->shape())."]\n";
//echo "axis=$axis,numClass=$numClass\n";
        if($axis===null) {
            $postfixShape = $A->shape();
            $prefixShape = $X->shape();
            //$numClass
            $ndimX = $X->ndim();
            $tmpShape = [];
            for($i=0;$i<$ndimX;$i++) {
                $tmpShape[] = array_shift($postfixShape);
            }
            if($tmpShape!=$prefixShape) {
                throw new InvalidArgumentException('Unmatch Shape:'.
                                        $this->printableShapes([$X,$A]));
            }
            $n = array_product($prefixShape);
            $k = array_product($postfixShape);
            $m = 1;
            $expandDims = false;
            $outputShape = array_merge([$numClass],$postfixShape);
        } else {
            $ndim = $A->ndim();
            $orgAxis = $axis;
            if($axis<0) {
                $axis = $ndim+$axis;
            }
            //if($axis<0 || $axis>$ndim-1) {
            //    throw new InvalidArgumentException("Invalid axis: ".$orgAxis);
            //}
            $postfixShape = $A->shape();
            $postfixX = $X->shape();
            if($postfixShape!=$postfixX) {
                throw new InvalidArgumentException('Unmatch Shape:'.
                                        $this->printableShapes([$X,$A]));
            }
            $prefixShape = [];
            for($i=0;$i<$axis;$i++) {
                $prefixShape[] = array_shift($postfixShape);
                array_shift($postfixX);
            }
            $m = array_product($prefixShape);
            $n = array_product($postfixShape);
            $k = 1;
            $expandDims = true;
            $outputShape = array_merge($prefixShape,[$numClass],$postfixShape);
        }
//echo "outputShape=[".implode(',',$outputShape)."]\n";
        if($dtype===null) {
            $dtype = $A->dtype();
        }
        if($B==null) {
            $B = $this->alloc($outputShape,$dtype);
            $this->zeros($B);
        } else {
            if($B->shape()!=$outputShape) {
                $shapeError = '('.implode(',',$A->shape()).'),('.implode(',',$B->shape()).')';
                throw new InvalidArgumentException("Unmatch shape of dimension: ".$shapeError);
            }
        }

        $AA = $A->buffer();
        $offA = $A->offset();
        $XX = $X->buffer();
        $offX = $X->offset();
        $BB = $B->buffer();
        $offB = $B->offset();

        if($expandDims) {
            return [ $reduce=true,
                $reverse=true,
                $addMode=false,
                $m,
                $n,
                $numClass,
                $XX,$offX,
                $BB,$offB,
                $AA,$offA];

        } else {
            return [ $reduce=false,
                $reverse=true,
                $addMode=false,
                $n,
                $k,
                $numClass,
                $XX,$offX,
                $BB,$offB,
                $AA,$offA];
        }
    }

/*
    public function translate_gatherb(
        NDArray $A,
        NDArray $X,
        int $axis=null,
        NDArray $output=null,
        bool $reverse=null,
        ) : array
    {
//echo "shapeX=[".implode(',',$X->shape())."],shapeA=[".implode(',',$A->shape())."]\n";
        $axis ??= 0;
        $ndimBatch = $X->ndim();
        $shape = $A->shape();
        $batchShape = array_slice($shape,0,$ndimBatch);
        if($batchShape!=$X->shape()) {
            throw new InvalidArgumentException('Unmatch Shape A and X:'.
                                    $this->printableShapes([$A,$X]));
        }
        $ndim = $A->ndim()-$ndimBatch;
        $orgAxis = $axis;
        if($axis<0) {
            $axis = $ndim+$axis;
        }
        if($axis<0||$axis>=$ndim) {
            throw new InvalidArgumentException('Unmatch Shape A and X and axis: '.
                                    $this->printableShapes([$A,$X]).' and axis is '.$orgAxis);
        }
        $prefixShape = array_slice($shape,$ndimBatch,$axis);
        $numClass = array_slice($shape,$ndimBatch+$axis,1)[0];
        $postfixShape = array_slice($shape,$ndimBatch+$axis+1);
        
        $m = array_product($batchShape);
        $n = array_product($prefixShape);
        $k = array_product($postfixShape);
        $outputShape = array_merge($batchShape,$prefixShape,$postfixShape);
        $dtype = $A->dtype();
        if($output==null) {
            $output = $this->zeros($outputShape,dtype:$dtype);
        } else {
            if($output->shape()!=$outputShape) {
                throw new InvalidArgumentException("Unmatch output shape of dimension: ".
                                            $this->printableShapes([$outputShape,$output]));
            }
        }

        $reverse ??= false;
        $addMode = false;
        $AA = $A->buffer();
        $offA = $A->offset();
        $XX = $X->buffer();
        $offX = $X->offset();
        $BB = $output->buffer();
        $offB = $output->offset();

        return [
            $reverse,
            $addMode,
            $m,
            $n,
            $k,
            $numClass,
            $AA,$offA,
            $XX,$offX,
            $BB,$offB,
        ];
    }

    public function translate_scatterb(
        NDArray $X,
        NDArray $A,
        int $numClass,
        int $axis=null,
        NDArray $output=null,
        ) : array
    {
        if($output===null) {
            $shape = $A->shape();
            $ndimBatch = $X->ndim();
            $batchShape = array_splice($shape,0,$ndimBatch);
            $prefixShape = array_splice($shape,0,$axis);
            $outputShape = array_merge($batchShape,$prefixShape,[$numClass],$shape);
            $output = $this->alloc($outputShape,dtype:$A->dtype());
        }
        return $this->translate_gatherb($output,$X,$axis,$A,$reverse);
    }
*/
    public function doGatherND(
        bool $reverse,
        bool $addMode,
        NDArray $A, 
        NDarray $X,
        ?int $batchDims=null,
        ?NDArray $B=null,
    ) : array
    {
        $batchDims ??= 0;
        //echo "params  sh: ".$this->printableShapes($A->shape())."\n";
        //echo "indices sh: ".$this->printableShapes($X->shape())."\n";
        //echo "batchDims: ".$batchDims."\n";
        $batchShape = $X->shape();
        $indexDepth = array_pop($batchShape);
        $outerShape = array_splice($batchShape,$batchDims);

        $innerShape = $A->shape();
        $paramsBatchShape = array_splice($innerShape,0,$batchDims);
        if($paramsBatchShape!=$batchShape) {
            throw new InvalidArgumentException("Unmatch batch shape of params and indices: ".
                $this->printableShapes([$paramsBatchShape,$batchShape]));
        }
        if($indexDepth > $A->ndim()-$batchDims) {
            throw new InvalidArgumentException(
                "ndim of params must greater than index depth or equal: ".
                "ndim of params gives ".$A->ndim()." , index depth gives ".$indexDepth);
        }
        $paramShape = array_splice($innerShape,0,$indexDepth);
        $outputShape = array_merge($batchShape,$outerShape,$innerShape);
        $m = array_product($batchShape);
        $n = array_product($outerShape);
        $k = array_product($innerShape);
        //echo "batchShape: ".$this->printableShapes($batchShape)."\n";
        //echo "outerShape: ".$this->printableShapes($outerShape)."\n";
        //echo "innerShape: ".$this->printableShapes($innerShape)."\n";
        //echo "indexDepth: ".$indexDepth."\n";
        //echo "outputShape: ".$this->printableShapes($outputShape)."\n";
        if($B==null) {
            $B = $this->zeros($outputShape,dtype:$A->dtype());
        } else {
            if($B->shape()!=$outputShape) {
                throw new InvalidArgumentException("Unmatch output shape: ".
                    $this->printableShapes([$outputShape]));
            }
        }
        $paramShape = $this->array($paramShape,dtype:NDArray::int32);
        $paramShape = $paramShape->buffer();
        $AA = $A->buffer();
        $offsetA = $A->offset();
        $XX = $X->buffer();
        $offsetX = $X->offset();
        $BB = $B->buffer();
        $offsetB = $B->offset();
        return [
            $reverse,
            $addMode,
            $m,
            $n,
            $k,
            $indexDepth,
            $paramShape, // paramShape[indexDepth]
            $AA, $offsetA,
            $XX, $offsetX,
            $BB, $offsetB,
        ];
    }

    /**
     * params:  (m, (indices(m,n)), k)
     * indices: (m, n, index_depth)
     * outputs: (m, n, k)
     */
    public function translate_gatherND(
        NDArray $params, 
        NDarray $indices,
        ?int $batchDims=null,
        ?NDArray $outputs=null,
    ) : array
    {
        return $this->doGatherND(
            $reverse=false,
            $addMode=false,
            $params,
            $indices,
            $batchDims,
            $outputs,
        );
    }

    /**
     * params:  (m, n, k)
     * indices: (m, n, index_depth)
     * outputs: (m, (indices(m,n)), k)
     */
    public function translate_scatterND(
        NDarray $indices,
        NDArray $updates,
        array $shape,
        ?int $batchDims=null,
        ?NDArray $outputs=null,
    ) : array
    {
        if($outputs==null) {
            $outputs = $this->zeros($shape,$updates->dtype());
        }
        
        return $this->doGatherND(
            $reverse=true,
            $addMode=false,
            $outputs,
            $indices,
            $batchDims,
            $updates,
        );
    }

    /**
     * params:  (m, n, k)
     * indices: (m, n, index_depth)
     * outputs: (m, (indices(m,n)), k)
     */
    public function translate_scatterNDAdd(
        NDarray $indices,
        NDArray $updates,
        array $shape,
        ?int $batchDims=null,
        ?NDArray $outputs=null,
    ) : array
    {
        if($outputs==null) {
            $outputs = $this->zeros($shape,$updates->dtype());
        }
        
        return $this->doGatherND(
            $reverse=true,
            $addMode=true,
            $outputs,
            $indices,
            $batchDims,
            $updates,
        );
    }

    /**
     * params:  (batches, m, numClass, k, len)
     * indices: (batches, n, k)
     * outputs: (batches, m, n, k, len)
     * outputs(batches, m, n, k, len) := A(batches, m, X(batches, n, k), k, len)
     */
    public function doGatherb(
        bool $reverse,
        bool $addMode,
        NDArray $A,
        NDarray $X,
        ?int $axis=null,
        ?int $batchDims=null,
        ?int $detailDepth=null,
        ?int $indexDepth=null,
        ?NDArray $B=null,
    ) : array {
        $batchDims ??= 0;
        if($batchDims<0) {
            $batchDims += $A->ndim();
        }
        if($batchDims>=$A->ndim()) {
            throw new InvalidArgumentException(
                "batchDims ($batchDims) must be less than to ndims of A (".$A->ndim().")"
            );
        }
        $axis ??= $batchDims;
        if($axis<0) {
            $axis += $A->ndim();
        }
        if($axis>=$A->ndim()) {
            throw new InvalidArgumentException(
                "axis ($axis) must be less than to ndims of A (".$A->ndim().")"
            );
        }
        $detailDepth ??= $axis+1;
        $indexDepth ??= $X->ndim();
        if($batchDims>$axis) {
            if($axis==0) {
                $batchDims = 0;
            } else {
                throw new InvalidArgumentException("batchDims ($batchDims) must be less than or equal to axis ($axis)");
            }
        }

        //echo "params  sh: ".$this->printableShapes($A->shape())."\n";
        //echo "indices sh: ".$this->printableShapes($X->shape())."\n";
        //echo "batchDims: ".$batchDims."\n";
        // A
        $batchShape = $A->shape();
        $outerShape = array_splice($batchShape, $batchDims);
        $innerShape = array_splice($outerShape, $axis-$batchDims);
        $numClass = array_shift($innerShape);
        $detailShape = array_splice($innerShape, $detailDepth-1-$axis);
        // X
        $batchShapeX = $X->shape();
        $indexShape = array_splice($batchShapeX, $batchDims);
        $innerShapeX = array_splice($indexShape, $indexDepth-$batchDims);

        //echo "==========\n";
        //echo "params:     ".$this->printableShapes($A->shape())."\n";
        //echo "indices:    ".$this->printableShapes($X->shape())."\n";
        //echo "batchDims:  $batchDims\n";
        //echo "axis:       $axis\n";
        //echo "detailDepth:$detailDepth\n";
        //echo "indexDepth: $indexDepth\n";
        //echo "==params==\n";
        //echo "batchShape: ".$this->printableShapes($batchShape)."\n";
        //echo "outerShape: ".$this->printableShapes($outerShape)."\n";
        //echo "numClass:   ".$numClass."\n";
        //echo "innerShape: ".$this->printableShapes($innerShape)."\n";
        //echo "detailShape: ".$this->printableShapes($detailShape)."\n";
        //echo "==indices==\n";
        //echo "batchShape: ".$this->printableShapes($batchShapeX)."\n";
        //echo "indexShape: ".$this->printableShapes($indexShape)."\n";
        //echo "innerShape: ".$this->printableShapes($innerShapeX)."\n";
        if($batchShape!=$batchShapeX) {
            throw new InvalidArgumentException(
                "Unmatch batch shape of params and indices: ".
                $this->printableShapes($batchShape).",".
                $this->printableShapes($batchShapeX).","
            );
        }
        if($innerShape!=$innerShapeX) {
            throw new InvalidArgumentException(
                "Unmatch inner shape of params and indices: ".
                "param's inner shape is ".$this->printableShapes($innerShape).",".
                "index's inner shape is ".$this->printableShapes($innerShapeX).","
            );
        }

        $batches = array_product($batchShape);
        $m = array_product($outerShape);
        $n = array_product($indexShape);
        $k = array_product($innerShape);
        $len = array_product($detailShape);
        $outputShape = array_merge($batchShape, $outerShape, $indexShape, $innerShape, $detailShape);
        //echo "==outputs==\n";
        //echo "outputShape: ".$this->printableShapes($outputShape)."\n";
        //echo "batches=$batches,m=$m,n=$n,k=$k,len=$len,numClass=$numClass\n";
        if($B==null) {
            $B = $this->zeros($this->alloc($outputShape, dtype:$A->dtype()));
        } else {
            if($B->shape()!=$outputShape) {
                $expects = $this->printableShapes($outputShape);
                $gives = $this->printableShapes($B->shape());
                if($reverse) {
                    $msg = "Unmatch updates shape: expects {$expects}, but gives {$gives}";
                } else {
                    $msg = "Unmatch output shape: expects {$expects}, but gives {$gives}";
                }
                throw new InvalidArgumentException($msg);
            }
        }
        $AA = $A->buffer();
        $offsetA = $A->offset();
        $XX = $X->buffer();
        $offsetX = $X->offset();
        $BB = $B->buffer();
        $offsetB = $B->offset();
        return [
            $reverse,
            $addMode,
            $batches,
            $m,
            $n,
            $k,
            $len,
            $numClass,
            $AA,
            $offsetA,
            $XX,
            $offsetX,
            $BB,
            $offsetB,
        ];
    }

    /**
     * params: (batches, m, numClass, k, len)
     * indices: (batches, n, k)
     * outputs: (batches, m, n, k, len)
     * outputs(batches, m, n, k, len) := A(batches, m, X(batches, n, k), k, len)
     */
    public function translate_gatherb(
        NDArray $params,
        NDarray $indices,
        ?int $axis=null,
        ?int $batchDims=null,
        ?int $detailDepth=null,
        ?int $indexDepth=null,
        ?NDArray $outputs=null,
    ) : array {
        return $this->doGatherb(
            $reverse=false,
            $addMode=false,
            $params,
            $indices,
            $axis,
            $batchDims,
            $detailDepth,
            $indexDepth,
            $outputs,
        );
    }

    /**
     * indices: (batches, n, k)
     * updates: (batches, m, n, k, len)
     * outputs: (batches, m, numClass, k, len)
     * outputs(batches, m, n, k, len) := A(batches, m, X(batches, n, k), k, len)
    */
    public function translate_scatterb(
        NDarray $indices,
        NDArray $updates,
        array $shape,
        ?int $axis=null,
        ?int $batchDims=null,
        ?int $detailDepth=null,
        ?int $indexDepth=null,
        ?NDArray $outputs=null,
    ): array
    {
        if($outputs==null) {
            $outputs = $this->zeros($this->alloc($shape,$updates->dtype()));
        }
        return $this->doGatherb(
            $reverse=true,
            $addMode=false,
            $outputs,
            $indices,
            $axis,
            $batchDims,
            $detailDepth,
            $indexDepth,
            $updates,
        );
        //return $outputs;
    }

    /**
     * indices: (batches, n, k)
     * updates: (batches, m, n, k, len)
     * outputs: (batches, m, numClass, k, len)
     * outputs(batches, m, n, k, len) := A(batches, m, X(batches, n, k), k, len)
    */
    public function translate_scatterbAdd(
        NDarray $indices,
        NDArray $updates,
        array $shape,
        ?int $axis=null,
        ?int $batchDims=null,
        ?int $detailDepth=null,
        ?int $indexDepth=null,
        ?NDArray $outputs=null,
    ) : array
    {
        if($outputs==null) {
            $outputs = $this->zeros($this->alloc($shape,$updates->dtype()));
        }
        return $this->doGatherb(
            $reverse=true,
            $addMode=true,
            $outputs,
            $indices,
            $axis,
            $batchDims,
            $detailDepth,
            $indexDepth,
            $updates,
        );
        //return $outputs;
    }

    public function translate_slice(
        bool $reverse,
        NDArray $input,
        array $begin,
        array $size,
        ?NDArray $output=null
        ) : array
    {
        if(!$reverse){
            $messageInput='Input';
        } else {
            $messageInput='Output';
        }
        $orgBegin = $begin;
        $orgSize = $size;
        $ndimBegin = count($begin);
        if($ndimBegin<1||$ndimBegin>3) {
            throw new InvalidArgumentException('begin must has 1 or 2 or 3 integer.');
        }
        $ndimSize = count($size);
        if($ndimSize<1||$ndimSize>3) {
            throw new InvalidArgumentException('Size must has 1 or 2 or 3 integer.');
        }
        if($ndimBegin!=$ndimSize){
            throw new InvalidArgumentException('Unmatch shape of begin and size');
        }
        $ndimInput = $input->ndim();
        if($ndimInput<$ndimBegin){
            throw new InvalidArgumentException($messageInput.' shape rank is low to slice');
        }
        $shape = $input->shape();

        // ndim = 0
        $m = array_shift($shape);
        $startAxis0 = array_shift($begin);
        if($startAxis0<0){
            $startAxis0 = $m+$startAxis0;
        }
        if($startAxis0<0||$startAxis0>=$m){
            throw new InvalidArgumentException('start of axis 0 is invalid value.');
        }
        $sizeAxis0 = array_shift($size);
        if($sizeAxis0<0){
            $sizeAxis0 = $m-$startAxis0+$sizeAxis0+1;
        }
        if($sizeAxis0<1||$startAxis0+$sizeAxis0>$m){
            throw new InvalidArgumentException('size of axis 0 is invalid value.');
        }

        // ndim = 1
        if($ndimBegin<=1){
            $n = 1;
            $startAxis1 = 0;
            $sizeAxis1 = 1;
        } else {
            $n = array_shift($shape);
            $startAxis1 = array_shift($begin);
            if($startAxis1<0){
                $startAxis1 = $n+$startAxis1;
            }
            if($startAxis1<0||$startAxis1>=$n){
                throw new InvalidArgumentException('start of axis 1 is invalid value.:begin=['.implode(',',$orgBegin).']');
            }
            $sizeAxis1 = array_shift($size);
            if($sizeAxis1<0){
                $sizeAxis1 = $n-$startAxis1+$sizeAxis1+1;
            }
            if($sizeAxis1<1||$startAxis1+$sizeAxis1>$n){
                throw new InvalidArgumentException('size of axis 1 is invalid value.');
            }
        }

        // ndim = 2
        if($ndimBegin<=2){
            $k = 1;
            $startAxis2 = 0;
            $sizeAxis2 = 1;
        } else {
            $k = array_shift($shape);
            $startAxis2 = array_shift($begin);
            if($startAxis2<0){
                $startAxis2 = $k+$startAxis2;
            }
            if($startAxis2<0||$startAxis2>=$k){
                throw new InvalidArgumentException('start of axis 2 is invalid value.:begin=['.implode(',',$orgBegin).']');
            }
            $sizeAxis2 = array_shift($size);
            if($sizeAxis2<0){
                $sizeAxis2 = $k-$startAxis2+$sizeAxis2+1;
            }
            if($sizeAxis2<1||$startAxis2+$sizeAxis2>$k){
                throw new InvalidArgumentException('size of axis 2 is invalid value.');
            }
        }
        $itemSize = array_product($shape);
        $outputShape = [$sizeAxis0];
        if($ndimBegin>=2){
            array_push($outputShape,
                $sizeAxis1);
        }
        if($ndimBegin>=3){
            array_push($outputShape,
                $sizeAxis2);
        }
        $outputShape = array_merge(
            $outputShape,$shape);
        if($output==null){
            $output = $this->alloc($outputShape,$input->dtype());
        }else{
            if($outputShape!=$output->shape()){
                throw new InvalidArgumentException('Unmatch output shape: '.
                    $this->printableShapes($outputShape).'<=>'.
                    $this->printableShapes($output->shape()));
            }
        }

        $A = $input->buffer();
        $offsetA = $input->offset();
        $Y = $output->buffer();
        $offsetY = $output->offset();
        $incA = 1;
        $incY = 1;
        return [
            $reverse,
            $addMode=false,
            $m,
            $n,
            $k,
            $itemSize,
            $A,$offsetA,$incA,
            $Y,$offsetY,$incY,
            $startAxis0,$sizeAxis0,
            $startAxis1,$sizeAxis1,
            $startAxis2,$sizeAxis2
        ];
    }

    public function translate_repeat(
        NDArray $A, int $repeats, ?int $axis=null, ?bool $keepdims=null,
        ?NDArray $output=null)
    {
        if($repeats<1) {
            throw new InvalidArgumentException('repeats argument must be one or greater.');
        }
        if($axis!==null) {
            $ndim = $A->ndim();
            if($axis<0) {
                $axis = $ndim+$axis;
            }
            if($A->ndim()<$axis) {
                throw new InvalidArgumentException('dimension rank must be two or greater.');
            }
        }
        $innerShape = $A->shape();
        $outerShape = [];
        if($axis!==null) {
            for($i=0;$i<$axis;$i++) {
                $outerShape[] = array_shift($innerShape);
            }
        }
        $base = 1;
        if($axis===null) {
            $outputShape = [(int)array_product(
                    array_merge($outerShape,[$repeats],$innerShape))];
        } else {
            if($keepdims) {
                $base = array_shift($innerShape);
                if($base===null) {
                    throw new InvalidArgumentException('dimension rank must be two or greater on keepdims.');
                }
                $outputShape = array_merge($outerShape,[$repeats*$base],$innerShape);
            } else {
                $outputShape = array_merge($outerShape,[$repeats],$innerShape);
            }
        }
        if($output==null) {
            $B = $this->zeros($outputShape,$A->dtype());
        } else {
            if($output->shape()!=$outputShape || $output->dtype()!=$A->dtype()) {
                throw new InvalidArgumentException('Unmatch output shape');
            }
            $B = $output;
        }
        $m = (int)array_product($outerShape);
        $k = (int)array_product($innerShape)*$base;
        $AA = $A->buffer();
        $offA = $A->offset();
        $BB = $B->buffer();
        $offB = $B->offset();
        return [
            $m,
            $k,
            $repeats,
            $AA,$offA,
            $BB,$offB
        ];
    }

    public function translate_onehot(
        NDArray $X,
        int $numClass,
        ?float $a=null,
        ?NDArray $Y=null) : array
    {
        if($X->ndim()!=1) {
            throw new InvalidArgumentException('"X" must be 1D-NDArray.');
        }
        $sizeX = $X->size();
        if($Y===null) {
            $Y = $this->zeros($this->alloc([$sizeX,$numClass]));
        }
        if($Y->ndim()!=2) {
            throw new InvalidArgumentException('"Y" must be 2D-NDArray.');
        }
        [$m,$n] = $Y->shape();
        if($m!=$sizeX || $n!=$numClass) {
            $shapeError = '('.implode(',',$X->shape()).'),('.implode(',',$Y->shape()).')';
            throw new InvalidArgumentException('Unmatch shape of dimension "X" and "Y" and "numClass": '.$shapeError);
        }
        if($a===null) {
            $a = 1.0;
        }
        $XX = $X->buffer();
        $offX = $X->offset();
        $YY = $Y->buffer();
        $offY = $Y->offset();
        $ldY = $n;

        return [
            $m,
            $n,
            $a,
            $XX,$offX,1,
            $YY,$offY,$ldY
        ];
    }

    public function translate_reduceSum(
        NDArray $A,
        ?int $axis=null,
        ?NDArray $B=null,
        ?int $dtype=null) : array
    {
        $ndim = $A->ndim();
        if($axis<0) {
            $axis = $ndim+$axis;
        }
        if($axis<0 || $axis>$ndim-1) {
            throw new InvalidArgumentException("Invalid axis");
        }
        $postfixShape = $A->shape();
        $prefixShape = [];
        for($i=0;$i<$axis;$i++) {
            $prefixShape[] = array_shift($postfixShape);
        }
        $n = array_shift($postfixShape);
        $m = array_product($prefixShape);
        $k = array_product($postfixShape);
        $outputShape = array_merge($prefixShape,$postfixShape);
        if($dtype===null) {
            $dtype = $A->dtype();
        }
        if($B==null) {
            $B = $this->alloc($outputShape,$dtype);
        } else {
            if($B->shape()!=$outputShape) {
                $shapeError = '('.implode(',',$A->shape()).'),('.implode(',',$B->shape()).')';
                throw new InvalidArgumentException("Unmatch shape of dimension: ".$shapeError);
            }
        }
        $AA = $A->buffer();
        $offA = $A->offset();
        $BB = $B->buffer();
        $offB = $B->offset();
        return [
            $m,
            $n,
            $k,
            $AA,$offA,
            $BB,$offB
        ];
    }

    public function translate_astype(NDArray $X, $dtype, NDArray $Y) : array
    {
        $n = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();
        $YY = $Y->buffer();
        $offY = $Y->offset();
 
        return [
            $n,
            $dtype,
            $XX,$offX,1,
            $YY,$offY,1
        ];
    }
 
    public function translate_searchsorted(
        NDArray $A,
        NDArray $X,
        ?bool $right=null,
        ?int $dtype=null,
        ?NDArray $Y=null
        ) : array
    {
        if($A->ndim()==1) {
            $individual = false;
        } elseif($A->ndim()==2) {
            $individual = true;
        } else {
            throw new InvalidArgumentException('A must be 1D or 2D NDArray.');
        }
        if($right===null) {
            $right = false;
        }
        if($dtype===null) {
            $dtype = NDArray::uint32;
        }
        if($Y===null) {
            $Y = $this->alloc($X->shape(),$dtype);
        }
        $dtype = $Y->dtype();
        if($dtype!=NDArray::uint32&&$dtype!=NDArray::int32&&
            $dtype!=NDArray::uint64&&$dtype!=NDArray::int64) {
            throw new InvalidArgumentException('dtype of Y must be int32 or int64');
        }
        if($X->shape()!=$Y->shape()) {
            $shapeError = '('.implode(',',$X->shape()).'),('.implode(',',$Y->shape()).')';
            throw new InvalidArgumentException("Unmatch shape of dimension: ".$shapeError);
        }
        if($individual) {
            [$m,$n] = $A->shape();
            if($m!=$X->size()) {
                $shapeError = '('.implode(',',$A->shape()).'),('.implode(',',$X->shape()).')';
                throw new InvalidArgumentException("Unmatch shape of dimension A,X: ".$shapeError);
            }
            $ldA = $n;
        } else {
            $m = $X->size();
            $n = $A->size();
            $ldA = 0;
        }
        $AA = $A->buffer();
        $offA = $A->offset();
        $XX = $X->buffer();
        $offX = $X->offset();
        $YY = $Y->buffer();
        $offY = $Y->offset();
 
        return [
            $m,
            $n,
            $AA,$offA,$ldA,
            $XX,$offX,1,
            $right,
            $YY,$offY,1
        ];
    }

    public function translate_cumsum(
        NDArray $X,
        ?bool $exclusive=null,
        ?bool $reverse=null,
        ?NDArray $Y=null
        ) : array
    {
        if($exclusive===null) {
            $exclusive = false;
        }
        if($reverse===null) {
            $reverse = false;
        }
        if($Y===null) {
            $Y = $this->alloc($X->shape(),$X->dtype());
        }
        if($X->shape()!=$Y->shape()) {
            $shapeError = '('.implode(',',$X->shape()).'),('.implode(',',$Y->shape()).')';
            throw new InvalidArgumentException("Unmatch shape of dimension: ".$shapeError);
        }
        $n = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();
        $YY = $Y->buffer();
        $offY = $Y->offset();
 
        return [
            $n,
            $XX,$offX,1,
            $exclusive,
            $reverse,
            $YY,$offY,1
        ];
    }

    public function translate_cumsumb(
        NDArray $inputs,
        ?int $axis=null,
        ?bool $exclusive=null,
        ?bool $reverse=null,
        ?NDArray $outputs=null
        ) : array
    {
        $ndim = $inputs->ndim();
        $origAxis = $axis;
        $axis ??= 0;
        if($axis<0) {
            $axis = $ndim+$axis;
        }
        if($axis<0 || $axis>$ndim-1) {
            $origAxis = $origAxis ?? 'null';
            throw new InvalidArgumentException("Invalid axis: ".$origAxis);
        }
        $exclusive ??= false;
        $reverse ??= false;
        $postfixShape = $inputs->shape();
        $prefixShape = array_splice($postfixShape, 0, $axis);
        $m = array_product($prefixShape);
        $n = array_shift($postfixShape);
        $k = array_product($postfixShape);
        if($outputs===null) {
            $outputs = $this->alloc($inputs->shape(),dtype:$inputs->dtype());
        }
        if($inputs->shape()!=$outputs->shape()) {
            $shapeError = '('.implode(',',$inputs->shape()).'),('.implode(',',$outputs->shape()).')';
            throw new InvalidArgumentException("Unmatch shape of dimension: ".$shapeError);
        }
        $AA = $inputs->buffer();
        $offA = $inputs->offset();
        $BB = $outputs->buffer();
        $offB = $outputs->offset();

        return [
            $m,
            $n,
            $k,
            $AA,$offA,
            $exclusive,
            $reverse,
            $BB,$offB,
        ];
    }

    public function translate_transpose(
        NDArray $A,
        array|NDArray $perm,
        NDArray $B,
        ) : array
    {
        $AA = $A->buffer();
        $BB = $B->buffer();
        $offsetA = $A->offset();
        $offsetB = $B->offset();
        $sourceShape = $this->array($A->shape(),NDArray::int32)->buffer();
        if(is_array($perm)) {
            $perm = $this->array($perm,NDArray::int32);
        }
        $permBuf = $perm->buffer();

        return [
            $sourceShape,
            $permBuf,
            $AA, $offsetA,
            $BB, $offsetB,
        ];
    }

    public function translate_bandpart(
        NDArray $A,
        int $lower,
        int $upper,
    ) : array
    {
        if($A->ndim()<2) {
            throw new InvalidArgumentException('input array must be 2D or upper.');
        }
        $shape = $A->shape();
        $k = array_pop($shape);
        $n = array_pop($shape);
        $m = (int)array_product($shape);
        $buffer = $A->buffer();
        $offset = $A->offset();
        return [
            $m,$n,$k,
            $buffer,$offset,
            $lower,
            $upper,
        ];
    }

    public function translate_randomUniform(
        array $shape,
        $low,
        $high,
        ?int $dtype=null,
        ?int $seed=null,
        ?NDArray $output=null) : array
    {
        $X = $output;
        if($dtype!==null&&$X!==null) {
            if ($X->dtype()!=$dtype) {
                throw new InvalidArgumentException('Unmatch dtype and dtype of X');
            }
        }
        if($X===null) {
            $X = $this->alloc($shape,$dtype);
        } else {
            if ($X->shape()!=$shape) {
                throw new InvalidArgumentException('Unmatch shape and shape of X');
            }
            if(!is_numeric($low)||!is_numeric($high)){
                throw new InvalidArgumentException('low and high must be integer or float');
            }
        }
        if($seed===null) {
            $seed = random_int(~PHP_INT_MAX,PHP_INT_MAX);
        }

        $n = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();

        return [
            $n,
            $XX,$offX,1,
            $low,
            $high,
            $seed
        ];
    }

    public function translate_randomNormal(
        array $shape,
        $mean,
        $scale,
        ?int $dtype=null,
        ?int $seed=null,
        ?NDArray $output=null) : array
    {
        $X = $output;
        if($dtype!==null&&$X!==null) {
            if ($X->dtype()!=$dtype) {
                throw new InvalidArgumentException('Unmatch dtype and dtype of X');
            }
        }
        if($X===null) {
            $X = $this->zeros($shape,$dtype);
        } else {
            if ($X->shape()!=$shape) {
                throw new InvalidArgumentException('Unmatch shape and shape of X');
            }
            if(!is_numeric($mean)||!is_numeric($scale)){
                throw new InvalidArgumentException('mean and scale must be integer or float');
            }
        }
        if($seed===null) {
            $seed = random_int(~PHP_INT_MAX,PHP_INT_MAX);
        }

        $n = $X->size();
        $XX = $X->buffer();
        $offX = $X->offset();

        return [
            $n,
            $XX,$offX,1,
            $mean,
            $scale,
            $seed
        ];
    }
    
    public function translate_randomSequence(
        int $base,
        ?int $size=null,
        ?int $seed=null,
        ?int $dtype=null,
        ?NDArray $output=null,
        ) : array
    {
        $X = $output;
        if($size==null) {
            $size = $base;
        }
        if($X==null) {
            $dtype = $dtype ?? NDArray::int32;
            $X = $this->zeros([$base],$dtype);
        } else {
            $dtype = $dtype ?? $X->dtype();
            if($X->dtype()!=$dtype || $X->count()!=$base) {
                throw new InvalidArgumentException("output size must be the same of base and same dtype");
            }
        }
        if($seed===null) {
            $seed = random_int(~PHP_INT_MAX,PHP_INT_MAX);
        }

        $n = $base;
        $XX = $X->buffer();
        $offX = $X->offset();

        return [
            $n,
            $size,
            $XX,$offX,1,
            $seed
        ];
        //$X = $X[[0,$size-1]];
        //return $X;
    }

    public function translate_masking(
        NDArray $mask,
        NDArray $data,
        ?int $batchDims=null,
        ?int $axis=null,
        ?float $fill=null,
        ?int $mode=null
        ) : array
    {

        //if($mask->dtype()!=NDArray::bool) {
        //    $types = $this->dtypeToString[$mask->dtype()];
        //    throw new InvalidArgumentException('dtype of mask must be bool. :'.$types);
        //}
        $batchDims ??= 0;
        if($batchDims<0) {
            $batchDims += $data->ndim();
        }
        if($batchDims>$data->ndim()) {
            throw new InvalidArgumentException(
                "batchDims ($batchDims) must be less than dims of data (".$data->ndim().") or equal."
            );
        }
        $axis ??= $batchDims;
        if($axis<0) {
            $axis += $data->ndim();
        }
        if($axis>$data->ndim()) {
            throw new InvalidArgumentException(
                "axis ($axis) must be less than to ndims of data (".$data->ndim().")"
            );
        }
        if($batchDims>$axis) {
            if($axis==0) {
                $batchDims = 0;
            } else {
                throw new InvalidArgumentException("batchDims ($batchDims) must be less than or equal to axis ($axis)");
            }
        }
        if($mask->ndim()<$batchDims) {
            throw new InvalidArgumentException(
                "batchDims ($batchDims) must be less than or equal to ndims of mask (".$mask->ndim().")"
            );
        }
        $fill ??= 0;
        $mode ??= 0;

        //echo "data  sh: ".$this->printableShapes($data->shape())."\n";
        //echo "mask sh: ".$this->printableShapes($mask->shape())."\n";
        //echo "batchDims: ".$batchDims."\n";
        // data
        $outerShape = $data->shape();
        $brodacastShape = array_splice($outerShape, $batchDims);
        $innerShape = array_splice($brodacastShape, $axis-$batchDims);
        $innterBroadcastShape = array_splice($innerShape,$mask->ndim()-$batchDims);
        // mask
        $outerShapeX = $mask->shape();
        $innerShapeX = array_splice($outerShapeX, $batchDims);

        //echo "mask: ".$this->printableShapes($mask->shape())."\n";
        //echo "data: ".$this->printableShapes($data->shape())."\n";
        //echo "batchDims: ".$batchDims."\n";
        //echo "axis: ".$axis."\n";
        //echo "outerShape: ".$this->printableShapes($outerShape)."\n";
        //echo "brodacastShape: ".$this->printableShapes($brodacastShape)."\n";
        //echo "innerShape: ".$this->printableShapes($innerShape)."\n";
        //echo "outerShapeX: ".$this->printableShapes($outerShapeX)."\n";
        //echo "innerShapeX: ".$this->printableShapes($innerShapeX)."\n";

        if($outerShape!=$outerShapeX) {
            throw new InvalidArgumentException('Unmatch dimension outer shape.: '.
                'mask('.implode(',',$mask->shape()).') => data('.implode(',',$data->shape()).')');
        }
        if($innerShape!=$innerShapeX) {
            throw new InvalidArgumentException('Unmatch dimension inner shape.: '.
                'mask('.implode(',',$mask->shape()).') => data('.implode(',',$data->shape()).')');
        }
        $m = (int)array_product($outerShape);
        $n = (int)array_product($brodacastShape);
        $k = (int)array_product($innerShape);
        $len = (int)array_product($innterBroadcastShape);

        if($n==1) {
            $k = $m*$k;
            $m = 1;
        }
        if($k==1) {
            $k = $m;
            $len = $n*$len;
            $m = 1;
            $n = 1;
        }
        //echo "m=$m,n=$n,k=$k\n";

        $XX = $mask->buffer();
        $offX = $mask->offset();
        $AA = $data->buffer();
        $offA = $data->offset();

        //$this->math->masking(  // tempolary implements
        //    $m,
        //    $n,
        //    $k,
        //    $fill,
        //    $XX,$offX,
        //    $AA,$offA,
        //);
        //return $A;

        return [
            $m,
            $n,
            $k,
            $len,
            $fill,
            $mode,
            $XX,$offX,
            $AA,$offA,
        ];
    }
    
    protected function parseEinsum(
        string $equation,
        NDArray ...$arrays,
    ) : array
    {
        [$a,$b] = $arrays;
        $equation = str_replace(' ','',$equation);
        if(!preg_match(
            "/([a-zA-Z]+),([a-zA-Z]+)->([a-zA-Z]+)/",
            $equation,
            $split_string
        )) {
            throw new InvalidArgumentException("Not recognized as an equation: '{$equation}'");
        }
        $labelA = $split_string[1];
        $labelB = $split_string[2];
        $labelC = $split_string[3];
        if($a->ndim()!=strlen($labelA)) {
            $msg = "(".implode(',',$a->shape()).")";
            throw new InvalidArgumentException("The rank of array A does not match the equation: '{$equation}', but {$msg} given.");
        }
        if($b->ndim()!=strlen($labelB)) {
            $msg = "(".implode(',',$b->shape()).")";
            throw new InvalidArgumentException("The rank of array B does not match the equation: '{$equation}', but {$msg} given.");
        }

        $allIndices = [];
        $labels = [$labelA,$labelB];
        $shapes = [$a->shape(),$b->shape()];
        foreach($labels as $idx => $label) {
            foreach(str_split($label) as $axis => $chr) {
                if(array_key_exists($chr,$allIndices)) {
                    if($allIndices[$chr]!=$shapes[$idx][$axis]) {
                        $shapeError = '('.implode(',',$a->shape()).'),('.implode(',',$b->shape()).')';
                        throw new InvalidArgumentException("Unmatch shapes in index letter '{$chr}' in equation:'{$equation}', but shapes {$shapeError} given.");
                    }
                } else {
                    $allIndices[$chr] = $shapes[$idx][$axis];
                }
            }
        }
        $outputShape = [];
        foreach(str_split($labelC) as $axis => $chr) {
            if(!array_key_exists($chr,$allIndices)) {
                throw new InvalidArgumentException("Target index letter '{$chr}' not found in inputs.:'{$equation}'");
            }
            $outputShape[$chr] = $allIndices[$chr];
            unset($allIndices[$chr]);
        }
        $allIndices = array_merge($outputShape,$allIndices);
        $outputShape = array_values($outputShape);

        // complie labels
        $allLabels = array_keys($allIndices);
        $sizeOfIndices = array_values($allIndices);
        $labels = [$labelA,$labelB,$labelC];
        $compiledLabels = [];
        foreach($labels as $label) {
            $compiledLabel = [];
            foreach(str_split($label) as $axis => $chr) {
                $compiledLabel[] = array_search($chr,$allLabels);
            }
            $compiledLabels[] = $compiledLabel;
        }

        return [
            $sizeOfIndices,
            $compiledLabels,
            $outputShape,
        ];
    }

    public function einsum_build_lds(
        array $sizeOfIndices,
        array $label,
        array $shape,
    ) : array
    {
        $lds = array_fill(0,count($sizeOfIndices),0);
        $ld = 1;
        foreach(array_reverse($label,true) as $axis => $lbl) {
            $broadcast = false;
            if($shape[$axis]==1) {
                $broadcast = true;
            }
            if(!$broadcast) {
                $lds[$lbl] += $ld;
            }
            $ld *= $sizeOfIndices[$lbl];
        }
        return $lds;
    }

    public function translate_einsum(
        string $equation,
        NDArray $a,
        NDArray $b,
    ) : array
    {
        $shapeA = $a->shape();
        $shapeB = $b->shape();
        $cacheKey = $equation;
            [
                $sizeOfIndices,
                $compiledLabels,
                $outputShape,
            ] = $this->parseEinsum($equation, $a, $b);
            [$labelA,$labelB,$labelC] = $compiledLabels;
        $ldA = $this->einsum_build_lds($sizeOfIndices,$labelA,$shapeA);
        $ldB = $this->einsum_build_lds($sizeOfIndices,$labelB,$shapeB);
        $sizeOfIndices = $this->array($sizeOfIndices,dtype:NDArray::int32)->buffer();
        $ldA = $this->array($ldA,dtype:NDArray::int32)->buffer();
        $ldB = $this->array($ldB,dtype:NDArray::int32)->buffer();

        $outputs = $this->zeros($outputShape,dtype:$a->dtype());
        //$this->zeros($outputs);

        $AA = $a->buffer();
        $offsetA = $a->offset();
        $BB = $b->buffer();
        $offsetB = $b->offset();
        $CC = $outputs->buffer();
        $offsetC = $outputs->offset();
        $ndimC = $outputs->ndim();

        return [
            $sizeOfIndices,
            $AA,
            $offsetA,
            $ldA,
            $BB,
            $offsetB,
            $ldB,
            $CC,
            $offsetC,
            $ndimC,
            $outputs,
        ];
    }

    public function translate_einsum4p1(
        string $equation,
        NDArray $a,
        NDArray $b,
        ?NDArray $c=null,
    ) : array
    {
        $shapeA = $a->shape();
        $shapeB = $b->shape();

            [
                $dims,
                $compiledLabels,
                $shapeC,
            ] = $this->parseEinsum($equation, $a, $b);
            [$labelA,$labelB,$labelC] = $compiledLabels;
            $sumDims = count($dims)-count($labelC);
            $inputMaxDims = 5-(1-$sumDims);
            if(count($labelC)>4) {
                throw new InvalidArgumentException("rank of C must be less than 4D or equal. (".implode(',',($dims?$dims:array_slice($dims,0,-$dims))).") given.");
            }
            if($sumDims>1) {
                throw new InvalidArgumentException("rank of overlay dims must be less than 1D or equal. '{$equation}' and {$sumDims} sum dims given.");
            }
            if(count($dims)>$inputMaxDims) {
                throw new InvalidArgumentException("dims must be less than {$inputMaxDims}D or equal.");
            }
            if(count($labelA)>$inputMaxDims) {
                throw new InvalidArgumentException("rank of A must be less than {$inputMaxDims}D or equal. (".implode(',',$a->shape()).") given.");
            }
            if(count($labelB)>$inputMaxDims) {
                throw new InvalidArgumentException("rank of B must be less than {$inputMaxDims}D or equal. (".implode(',',$b->shape()).") given.");
            }

        $ldA = $this->einsum_build_lds($dims,$labelA,$shapeA);
        $ldB = $this->einsum_build_lds($dims,$labelB,$shapeB);
    
        if($c==null) {
            $c = $this->zeros($shapeC,dtype:$a->dtype());
        } else {
            if($c->shape()!=$shapeC) {
                throw new InvalidArgumentException(
                    "unmatch shape of C. the shape must be (".implode(',',$shapeC)."). ".
                    "(".implode(',',$c->shape()).") given.");
            }
        }
        if($sumDims==0) {
            $dims[] = 1;
            $ldA[]  = 1;
            $ldB[]  = 1;
        }
        //echo "equation=$equation\n";
        //echo "labelA=(".implode(',',$labelA).")\n";
        //echo "labelB=(".implode(',',$labelB).")\n";
        //echo "labelC=(".implode(',',$labelC).")\n";
        //echo "shapeC=(".implode(',',$shapeC).")\n";
        //echo "dims=(".implode(',',$dims).")\n";
        //echo "ldA=(".implode(',',$ldA).")\n";
        //echo "ldB=(".implode(',',$ldB).")\n";
        $dims = array_pad($dims,-5,1);
        $ldA = array_pad($ldA,-5,0);
        $ldB = array_pad($ldB,-5,0);
        //echo "dims=(".implode(',',$dims).")\n";
        //echo "ldA=(".implode(',',$ldA).")\n";
        //echo "ldB=(".implode(',',$ldB).")\n";
        [$dim0,$dim1,$dim2,$dim3,$dim4] = $dims;
        [$ldA0,$ldA1,$ldA2,$ldA3,$ldA4] = $ldA;
        [$ldB0,$ldB1,$ldB2,$ldB3,$ldB4] = $ldB;
        $AA = $a->buffer();
        $offsetA = $a->offset();
        $BB = $b->buffer();
        $offsetB = $b->offset();
        $CC = $c->buffer();
        $offsetC = $c->offset();
        return [
            $dim0,
            $dim1,
            $dim2,
            $dim3,
            $dim4,
            $AA,
            $offsetA,
            $ldA0,
            $ldA1,
            $ldA2,
            $ldA3,
            $ldA4,
            $BB,
            $offsetB,
            $ldB0,
            $ldB1,
            $ldB2,
            $ldB3,
            $ldB4,
            $CC,
            $offsetC,
            $c,
        ];
    }

    public static function providerDtypesFloats()
    {
        return [
            'float32' => [[
                'dtype' => NDArray::float32,
            ]],
            'float64' => [[
                'dtype' => NDArray::float64,
            ]],
        ];
    }

    public static function providerDtypesFloatsAndInteger32()
    {
        return [
            'float32' => [[
                'dtype' => NDArray::float32,
            ]],
            'float64' => [[
                'dtype' => NDArray::float64,
            ]],
            'int32' => [[
                'dtype' => NDArray::int32,
            ]],
        ];
    }

    public static function providerDtypesFloatsAndInteger8()
    {
        return [
            'float32' => [[
                'dtype' => NDArray::float32,
            ]],
            'float64' => [[
                'dtype' => NDArray::float64,
            ]],
            'int32' => [[
                'dtype' => NDArray::int8,
            ]],
        ];
    }

    public static function providerDtypesFloatsAndAllInteger()
    {
        return [
            'float32' => [[
                'dtype' => NDArray::float32,
            ]],
            'float64' => [[
                'dtype' => NDArray::float64,
            ]],
            'int8' => [[
                'dtype' => NDArray::int8,
            ]],
            'uint8' => [[
                'dtype' => NDArray::uint8,
            ]],
            'int16' => [[
                'dtype' => NDArray::int16,
            ]],
            'uint16' => [[
                'dtype' => NDArray::uint16,
            ]],
            'int32' => [[
                'dtype' => NDArray::int32,
            ]],
            'uint32' => [[
                'dtype' => NDArray::uint32,
            ]],
            'int64' => [[
                'dtype' => NDArray::int64,
            ]],
            'uint64' => [[
                'dtype' => NDArray::uint64,
            ]],
        ];
    }

    public static function providerDtypesFloatsAndInteger3264()
    {
        return [
            'float32' => [[
                'dtype' => NDArray::float32,
            ]],
            'float64' => [[
                'dtype' => NDArray::float64,
            ]],
            'int32' => [[
                'dtype' => NDArray::int32,
            ]],
            'int64' => [[
                'dtype' => NDArray::int64,
            ]],
        ];
    }

    public static function providerDtypesFloatsAndInteger326w3246indexes()
    {
        return [
            'float32i32' => [[
                'dtype' => NDArray::float32,
                'indexdtype' => NDArray::int32,
            ]],
            'float64i32' => [[
                'dtype' => NDArray::float64,
                'indexdtype' => NDArray::int32,
            ]],
            'int32i32' => [[
                'dtype' => NDArray::int32,
                'indexdtype' => NDArray::int32,
            ]],
            'int64i32' => [[
                'dtype' => NDArray::int64,
                'indexdtype' => NDArray::int32,
            ]],
            'float32i64' => [[
                'dtype' => NDArray::float32,
                'indexdtype' => NDArray::int64,
            ]],
            'float64i64' => [[
                'dtype' => NDArray::float64,
                'indexdtype' => NDArray::int64,
            ]],
            'int32i64' => [[
                'dtype' => NDArray::int32,
                'indexdtype' => NDArray::int64,
            ]],
            'int64i64' => [[
                'dtype' => NDArray::int64,
                'indexdtype' => NDArray::int64,
            ]],
        ];
    }

    public function testGetNumThreads()
    {
        $matlib = $this->getMatlib();
        $n = $matlib->getNumThreads();
        $this->assertGreaterThan(0,$n);
    }

    public function testGetNumProcs()
    {
        $matlib = $this->getMatlib();
        $n = $matlib->getNumProcs();
        $this->assertGreaterThan(0,$n);
    }

    public function testGetParallel()
    {
        $matlib = $this->getMatlib();
        $n = $matlib->getParallel();

        if(PHP_OS=="WINNT") {
            //$this->assertEquals(Matlib::P_OPENMP,$n);
            $mt_enabled = ($n==Matlib::P_OPENMP || $n==Matlib::P_THREAD);
            $this->assertTrue($mt_enabled);
        } else {
            $mt_enabled = ($n==Matlib::P_OPENMP || $n==Matlib::P_THREAD);
            $this->assertTrue($mt_enabled);
        }
    }

    public function testGetVersion()
    {
        $matlib = $this->getMatlib();
        $s = $matlib->getVersion();
        $this->assertTrue(is_string($s));
    }

    public function testSumNormal()
    {
        $matlib = $this->getMatlib();
        $X = $this->array([100,-10,-1000],NDArray::float32);
        [$N,$XX,$offX,$incX] =
            $this->translate_amin($X);
        $min = $matlib->sum($N,$XX,$offX,$incX);
        $this->assertEquals(-910,$min);
 
        $X = $this->array([100,-10,-1000],NDArray::float64);
        [$N,$XX,$offX,$incX] =
            $this->translate_amin($X);
        $min = $matlib->sum($N,$XX,$offX,$incX);
        $this->assertEquals(-910,$min);
 
        $X = $this->array([-100,-100,-120],NDArray::int8);
        [$N,$XX,$offX,$incX] =
            $this->translate_amin($X);
        $min = $matlib->sum($N,$XX,$offX,$incX);
        $this->assertNotEquals(-320,$min);
        $this->assertEquals(-64,$min);
 
        $X = $this->array([10,20,30],NDArray::uint8);
        [$N,$XX,$offX,$incX] =
            $this->translate_amin($X);
        $min = $matlib->sum($N,$XX,$offX,$incX);
        $this->assertEquals(60,$min);
        $X = $this->array([-1,-2,-3],NDArray::uint8);
        [$N,$XX,$offX,$incX] =
            $this->translate_amin($X);
        $min = $matlib->sum($N,$XX,$offX,$incX);
        $this->assertNotEquals(256*3-1-2-3,$min);
        $this->assertEquals(250,$min);
 
        $X = $this->array([-100,-100,-120],NDArray::int16);
        [$N,$XX,$offX,$incX] =
            $this->translate_amin($X);
        $min = $matlib->sum($N,$XX,$offX,$incX);
        $this->assertEquals(-320,$min);
 
        $X = $this->array([10000,20000,30000],NDArray::uint16);
        [$N,$XX,$offX,$incX] =
            $this->translate_amin($X);
        $min = $matlib->sum($N,$XX,$offX,$incX);
        $this->assertEquals(60000,$min);
        $X = $this->array([-1,-2,-3],NDArray::uint16);
        [$N,$XX,$offX,$incX] =
            $this->translate_amin($X);
        $min = $matlib->sum($N,$XX,$offX,$incX);
        $this->assertNotEquals(65536*3-1-2-3,$min);
        $this->assertEquals(65530,$min);
 
        $X = $this->array([-100,-100,-120],NDArray::int32);
        [$N,$XX,$offX,$incX] =
            $this->translate_amin($X);
        $min = $matlib->sum($N,$XX,$offX,$incX);
        $this->assertEquals(-320,$min);
 
        $X = $this->array([100000000,200000000,300000000],NDArray::uint32);
        [$N,$XX,$offX,$incX] =
            $this->translate_amin($X);
        $min = $matlib->sum($N,$XX,$offX,$incX);
        $this->assertEquals(600000000,$min);
        $X = $this->array([-1,-2,-3],NDArray::uint32);
        [$N,$XX,$offX,$incX] =
            $this->translate_amin($X);
        $min = $matlib->sum($N,$XX,$offX,$incX);
        $this->assertNotEquals((2**32)*3-1-2-3,$min);
        $this->assertEquals(4294967290,$min);
 
        $X = $this->array([-100,-100,-120],NDArray::int64);
        [$N,$XX,$offX,$incX] =
            $this->translate_amin($X);
        $min = $matlib->sum($N,$XX,$offX,$incX);
        $this->assertEquals(-320,$min);
 
        $X = $this->array([-1,-2,-3],NDArray::uint64);
        [$N,$XX,$offX,$incX] =
            $this->translate_amin($X);
        $min = $matlib->sum($N,$XX,$offX,$incX);
        //$this->assertEquals((2**64)*3-1-2-3,$min);
        $this->assertEquals(-6,$min);
 
        $X = $this->array([true,false,true],NDArray::bool);
        [$N,$XX,$offX,$incX] =
            $this->translate_amin($X);
        $min = $matlib->sum($N,$XX,$offX,$incX);
        $this->assertEquals(2,$min);
    }
 
    public function testSumMinusN()
    {
        $matlib = $this->getMatlib();
 
        $X = $this->array([100,-10,1]);
        [$N,$XX,$offX,$incX] =
            $this->translate_amin($X);
 
        $N = 0;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument n must be greater than 0.');
        $min = $matlib->sum($N,$XX,$offX,$incX);
    }
 
    public function testSumMinusOffsetX()
    {
        $matlib = $this->getMatlib();
 
        $X = $this->array([100,-10,1]);
        [$N,$XX,$offX,$incX] =
            $this->translate_amin($X);
 
        $offX = -1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument offsetX must be greater than or equals 0.');
        $min = $matlib->sum($N,$XX,$offX,$incX);
    }
 
    public function testSumMinusIncX()
    {
        $matlib = $this->getMatlib();
 
        $X = $this->array([100,-10,1]);
        [$N,$XX,$offX,$incX] =
            $this->translate_amin($X);
 
        $incX = 0;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument incX must be greater than 0.');
        $min = $matlib->sum($N,$XX,$offX,$incX);
    }
 
    public function testSumIllegalBufferX()
    {
        $matlib = $this->getMatlib();
 
        $X = $this->array([100,-10,1]);
        [$N,$XX,$offX,$incX] =
            $this->translate_amin($X);
 
        $XX = new \stdClass();
        $this->expectException(TypeError::class);
        $this->expectExceptionMessage('must be of type Interop\Polite\Math\Matrix\LinearBuffer');
        $min = $matlib->sum($N,$XX,$offX,$incX);
    }
 
    public function testSumOverflowBufferXwithSize()
    {
        $matlib = $this->getMatlib();
 
        $X = $this->array([100,-10,1]);
        [$N,$XX,$offX,$incX] =
            $this->translate_amin($X);
 
        $XX = $this->array([100,-10])->buffer();
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Vector specification too large for bufferX');
        $min = $matlib->sum($N,$XX,$offX,$incX);
    }
 
    public function testSumOverflowBufferXwithOffsetX()
    {
        $matlib = $this->getMatlib();
 
        $X = $this->array([100,-10,1]);
        [$N,$XX,$offX,$incX] =
            $this->translate_amin($X);
 
        $offX = 1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Vector specification too large for bufferX');
        $min = $matlib->sum($N,$XX,$offX,$incX);
    }
 
    public function testSumOverflowBufferXwithIncX()
    {
        $matlib = $this->getMatlib();
 
        $X = $this->array([100,-10,1]);
        [$N,$XX,$offX,$incX] =
            $this->translate_amin($X);
 
        $incX = 2;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Vector specification too large for bufferX');
        $min = $matlib->sum($N,$XX,$offX,$incX);
    }
 
    #[DataProvider('providerDtypesFloats')]
    public function testMinNormal($params)
    {
        extract($params);
        $matlib = $this->getMatlib();
 
        $X = $this->array([100,-10,1],dtype:$dtype);
        [$N,$XX,$offX,$incX] =
            $this->translate_amin($X);
 
        $min = $matlib->imin($N,$XX,$offX,$incX);
        $this->assertEquals(1,$min);

        $X = $this->array([100,-10,1],dtype:NDArray::int32);
        [$N,$XX,$offX,$incX] =
            $this->translate_amin($X);
 
        $min = $matlib->imin($N,$XX,$offX,$incX);
        $this->assertEquals(1,$min);

        $X = $this->array([100,10,1],dtype:NDArray::uint64);
        [$N,$XX,$offX,$incX] =
            $this->translate_amin($X);
 
        $min = $matlib->imin($N,$XX,$offX,$incX);
        $this->assertEquals(2,$min);

    }
 
    public function testMinMinusN()
    {
        $matlib = $this->getMatlib();
 
        $X = $this->array([100,-10,1]);
        [$N,$XX,$offX,$incX] =
            $this->translate_amin($X);
 
        $N = 0;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument n must be greater than 0.');
        $min = $matlib->imin($N,$XX,$offX,$incX);
    }
 
    public function testMinMinusOffsetX()
    {
        $matlib = $this->getMatlib();
 
        $X = $this->array([100,-10,1]);
        [$N,$XX,$offX,$incX] =
            $this->translate_amin($X);
 
        $offX = -1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument offsetX must be greater than or equals 0.');
        $min = $matlib->imin($N,$XX,$offX,$incX);
    }
 
    public function testMinMinusIncX()
    {
        $matlib = $this->getMatlib();
 
        $X = $this->array([100,-10,1]);
        [$N,$XX,$offX,$incX] =
            $this->translate_amin($X);
 
        $incX = 0;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument incX must be greater than 0.');
        $min = $matlib->imin($N,$XX,$offX,$incX);
    }
 
    public function testMinIllegalBufferX()
    {
        $matlib = $this->getMatlib();
 
        $X = $this->array([100,-10,1]);
        [$N,$XX,$offX,$incX] =
            $this->translate_amin($X);
 
        $XX = new \stdClass();
        $this->expectException(TypeError::class);
        $this->expectExceptionMessage('must be of type Interop\Polite\Math\Matrix\LinearBuffer');
        $min = $matlib->imin($N,$XX,$offX,$incX);
    }
 
    public function testMinOverflowBufferXwithSize()
    {
        $matlib = $this->getMatlib();
 
        $X = $this->array([100,-10,1]);
        [$N,$XX,$offX,$incX] =
            $this->translate_amin($X);
 
        $XX = $this->array([100,-10])->buffer();
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Vector specification too large for bufferX');
        $min = $matlib->imin($N,$XX,$offX,$incX);
    }
 
    public function testMinOverflowBufferXwithOffsetX()
    {
        $matlib = $this->getMatlib();
 
        $X = $this->array([100,-10,1]);
        [$N,$XX,$offX,$incX] =
            $this->translate_amin($X);
 
        $offX = 1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Vector specification too large for bufferX');
        $min = $matlib->imin($N,$XX,$offX,$incX);
    }
 
    public function testMinOverflowBufferXwithIncX()
    {
        $matlib = $this->getMatlib();
 
        $X = $this->array([100,-10,1]);
        [$N,$XX,$offX,$incX] =
            $this->translate_amin($X);
 
        $incX = 2;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Vector specification too large for bufferX');
        $min = $matlib->imin($N,$XX,$offX,$incX);
    }
 
    #[DataProvider('providerDtypesFloats')]
    public function testMaxNormal($params)
    {
        extract($params);
        $matlib = $this->getMatlib();
 
        $X = $this->array([100,1000,-10,-1000],dtype:$dtype);
        [$N,$XX,$offX,$incX] =
            $this->translate_amin($X);
 
        $max = $matlib->imax($N,$XX,$offX,$incX);
        $this->assertEquals(1,$max);

        $X = $this->array([100,1000,-10,-1000],dtype:NDArray::int32);
        [$N,$XX,$offX,$incX] =
            $this->translate_amin($X);
 
        $max = $matlib->imax($N,$XX,$offX,$incX);
        $this->assertEquals(1,$max);

        $X = $this->array([100,1000, 0, 10],dtype:NDArray::uint64);
        [$N,$XX,$offX,$incX] =
            $this->translate_amin($X);
 
        $max = $matlib->imax($N,$XX,$offX,$incX);
        $this->assertEquals(1,$max);
    }
 
    public function testMaxMinusN()
    {
        $matlib = $this->getMatlib();
 
        $X = $this->array([100,-10,1]);
        [$N,$XX,$offX,$incX] =
            $this->translate_amin($X);
 
        $N = 0;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument n must be greater than 0.');
        $min = $matlib->imax($N,$XX,$offX,$incX);
    }
 
    public function testMaxMinusOffsetX()
    {
        $matlib = $this->getMatlib();
 
        $X = $this->array([100,-10,1]);
        [$N,$XX,$offX,$incX] =
            $this->translate_amin($X);
 
        $offX = -1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument offsetX must be greater than or equals 0.');
        $min = $matlib->imax($N,$XX,$offX,$incX);
    }
 
    public function testMaxMinusIncX()
    {
        $matlib = $this->getMatlib();
 
        $X = $this->array([100,-10,1]);
        [$N,$XX,$offX,$incX] =
            $this->translate_amin($X);
 
        $incX = 0;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument incX must be greater than 0.');
        $min = $matlib->imax($N,$XX,$offX,$incX);
    }
 
    public function testMaxIllegalBufferX()
    {
        $matlib = $this->getMatlib();
 
        $X = $this->array([100,-10,1]);
        [$N,$XX,$offX,$incX] =
            $this->translate_amin($X);
 
        $XX = new \stdClass();
        $this->expectException(TypeError::class);
        $this->expectExceptionMessage('must be of type Interop\Polite\Math\Matrix\LinearBuffer');
        $min = $matlib->imax($N,$XX,$offX,$incX);
    }
 
    public function testMaxOverflowBufferXwithSize()
    {
        $matlib = $this->getMatlib();
 
        $X = $this->array([100,-10,1]);
        [$N,$XX,$offX,$incX] =
            $this->translate_amin($X);
 
        $XX = $this->array([100,-10])->buffer();
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Vector specification too large for bufferX');
        $min = $matlib->imax($N,$XX,$offX,$incX);
    }
 
    public function testMaxOverflowBufferXwithOffsetX()
    {
        $matlib = $this->getMatlib();
 
        $X = $this->array([100,-10,1]);
        [$N,$XX,$offX,$incX] =
            $this->translate_amin($X);
 
        $offX = 1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Vector specification too large for bufferX');
        $min = $matlib->imax($N,$XX,$offX,$incX);
    }
 
    public function testMaxOverflowBufferXwithIncX()
    {
        $matlib = $this->getMatlib();
 
        $X = $this->array([100,-10,1]);
        [$N,$XX,$offX,$incX] =
            $this->translate_amin($X);
 
        $incX = 2;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Vector specification too large for bufferX');
        $min = $matlib->imax($N,$XX,$offX,$incX);
    }

    #[DataProvider('providerDtypesFloats')]
    public function testIncrementNormal($params)
    {
        extract($params);
        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3],dtype:$dtype);
        [$N,$alpha,$XX,$offX,$incX,$beta] =
            $this->translate_increment($X,10,2);

        $matlib->increment($N,$alpha,$XX,$offX,$incX,$beta);
        $this->assertEquals([12,14,16],$X->toArray());
    }

    public function testIncrementInvalidArgments()
    {
        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        [$N,$alpha,$XX,$offX,$incX,$beta] =
            $this->translate_increment($X,10,2);

        $this->expectException(TypeError::class);
        if(version_compare(PHP_VERSION, '8.0.0')<0) {
            $this->expectExceptionMessage('parameter 2 to be float');
        } else {
            $this->expectExceptionMessage('Argument #2 ($alpha) must be of type float');
        }
        $matlib->increment($N,new \stdClass(),$XX,$offX,$incX,$beta);
    }

    public function testIncrementMinusN()
    {
        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        [$N,$alpha,$XX,$offX,$incX,$beta] =
            $this->translate_increment($X,10,2);

        $N = 0;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument n must be greater than 0.');
        $matlib->increment($N,$alpha,$XX,$offX,$incX,$beta);
    }

    public function testIncrementMinusOffsetX()
    {
        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        [$N,$alpha,$XX,$offX,$incX,$beta] =
            $this->translate_increment($X,10,2);

        $offX = -1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument offsetX must be greater than or equals 0.');
        $matlib->increment($N,$alpha,$XX,$offX,$incX,$beta);
    }

    public function testIncrementMinusIncX()
    {
        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        [$N,$alpha,$XX,$offX,$incX,$beta] =
            $this->translate_increment($X,10,2);

        $incX = 0;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument incX must be greater than 0.');
        $matlib->increment($N,$alpha,$XX,$offX,$incX,$beta);
    }

    public function testIncrementIllegalBufferX()
    {
        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        [$N,$alpha,$XX,$offX,$incX,$beta] =
            $this->translate_increment($X,10,2);

        $XX = new \stdClass();
        $this->expectException(TypeError::class);
        $this->expectExceptionMessage('must be of type Interop\Polite\Math\Matrix\LinearBuffer');
        $matlib->increment($N,$alpha,$XX,$offX,$incX,$beta);
    }

    public function testIncrementOverflowBufferXwithSize()
    {
        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        [$N,$alpha,$XX,$offX,$incX,$beta] =
            $this->translate_increment($X,10,2);

        $XX = $this->array([1,2])->buffer();
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Vector specification too large for bufferX');
        $matlib->increment($N,$alpha,$XX,$offX,$incX,$beta);
    }

    public function testIncrementOverflowBufferXwithOffsetX()
    {
        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        [$N,$alpha,$XX,$offX,$incX,$beta] =
            $this->translate_increment($X,10,2);

        $offX = 1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Vector specification too large for bufferX');
        $matlib->increment($N,$alpha,$XX,$offX,$incX,$beta);
    }

    public function testIncrementOverflowBufferXwithIncX()
    {
        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        [$N,$alpha,$XX,$offX,$incX,$beta] =
            $this->translate_increment($X,10,2);

        $incX = 2;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Vector specification too large for bufferX');
        $matlib->increment($N,$alpha,$XX,$offX,$incX,$beta);
    }

    #[DataProvider('providerDtypesFloats')]
    public function testReciprocalNormal($params)
    {
        extract($params);
        $matlib = $this->getMatlib();

        $X = $this->array([3,2,0],dtype:$dtype);
        [$N,$alpha,$XX,$offX,$incX,$beta] =
            $this->translate_increment($X,4,-1);

        $matlib->reciprocal($N,$alpha,$XX,$offX,$incX,$beta);
        // X := 1 / ( alpha * X + beta )
        //    = [1/1, 1/2, 1/4]
        $this->assertEquals([1,0.5,0.25],$X->toArray());
    }

    public function testReciprocalZeroDivide()
    {
        $matlib = $this->getMatlib();

        $X = $this->array([4,2,0]);
        [$N,$alpha,$XX,$offX,$incX,$beta] =
            $this->translate_increment($X,$beta=0,$alpha=1);

        // X := 1 / ( alpha * X + beta )

        // *** CAUTION ***
        // disable checking for INFINITY values
        //$this->expectException(RuntimeException::class);
        //$this->expectExceptionMessage('Zero divide.');

        $matlib->reciprocal($N,$alpha,$XX,$offX,$incX,$beta);
        $this->assertEquals(
            [0.25,0.5,INF],
            $X->toArray());
    }

    public function testReciprocalMinusN()
    {
        $matlib = $this->getMatlib();

        $X = $this->array([3,2,0]);
        [$N,$alpha,$XX,$offX,$incX,$beta] =
            $this->translate_increment($X,4,-1);

        $N = 0;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument n must be greater than 0.');
        $matlib->reciprocal($N,$alpha,$XX,$offX,$incX,$beta);
    }

    public function testReciprocalMinusOffsetX()
    {
        $matlib = $this->getMatlib();

        $X = $this->array([3,2,0]);
        [$N,$alpha,$XX,$offX,$incX,$beta] =
            $this->translate_increment($X,4,-1);

        $offX = -1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument offsetX must be greater than or equals 0.');
        $matlib->reciprocal($N,$alpha,$XX,$offX,$incX,$beta);
    }

    public function testReciprocalMinusIncX()
    {
        $matlib = $this->getMatlib();

        $X = $this->array([3,2,0]);
        [$N,$alpha,$XX,$offX,$incX,$beta] =
            $this->translate_increment($X,4,-1);

        $incX = 0;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument incX must be greater than 0.');
        $matlib->reciprocal($N,$alpha,$XX,$offX,$incX,$beta);
    }

    public function testReciprocalIllegalBufferX()
    {
        $matlib = $this->getMatlib();

        $X = $this->array([3,2,0]);
        [$N,$alpha,$XX,$offX,$incX,$beta] =
            $this->translate_increment($X,4,-1);

        $XX = new \stdClass();
        $this->expectException(TypeError::class);
        $this->expectExceptionMessage('must be of type Interop\Polite\Math\Matrix\LinearBuffer');
        $matlib->reciprocal($N,$alpha,$XX,$offX,$incX,$beta);
    }

    public function testReciprocalOverflowBufferXwithSize()
    {
        $matlib = $this->getMatlib();

        $X = $this->array([3,2,0]);
        [$N,$alpha,$XX,$offX,$incX,$beta] =
            $this->translate_increment($X,4,-1);

        $XX = $this->array([3,2])->buffer();
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Vector specification too large for bufferX');
        $matlib->reciprocal($N,$alpha,$XX,$offX,$incX,$beta);
    }

    public function testReciprocalOverflowBufferXwithOffsetX()
    {
        $matlib = $this->getMatlib();

        $X = $this->array([3,2,0]);
        [$N,$alpha,$XX,$offX,$incX,$beta] =
            $this->translate_increment($X,4,-1);

        $offX = 1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Vector specification too large for bufferX');
        $matlib->reciprocal($N,$alpha,$XX,$offX,$incX,$beta);
    }

    public function testReciprocalOverflowBufferXwithIncX()
    {
        $matlib = $this->getMatlib();

        $X = $this->array([3,2,0]);
        [$N,$alpha,$XX,$offX,$incX,$beta] =
            $this->translate_increment($X,4,-1);

        $incX = 2;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Vector specification too large for bufferX');
        $matlib->reciprocal($N,$alpha,$XX,$offX,$incX,$beta);
    }

    #[DataProvider('providerDtypesFloats')]
    public function testMaximumNormal($params)
    {
        extract($params);
        $matlib = $this->getMatlib();

        $A = $this->array([[1,2],[2,3],[3,4]],dtype:$dtype);
        $X = $this->array([2,3],dtype:$dtype);
        [$M,$N,$AA,$offA,$ldA,$XX,$offX,$incX] =
            $this->translate_maximum($A,$X);

        $matlib->maximum($M,$N,$AA,$offA,$ldA,$XX,$offX,$incX);
        $this->assertEquals([[2,3],[2,3],[3,4]],$A->toArray());
    }

    public function testMaximumMinusM()
    {
        $matlib = $this->getMatlib();

        $A = $this->array([[1,2],[2,3],[3,4]]);
        $X = $this->array([2,3]);
        [$M,$N,$AA,$offA,$ldA,$XX,$offX,$incX] =
            $this->translate_maximum($A,$X);

        $M = 0;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument m must be greater than 0.');
        $matlib->maximum($M,$N,$AA,$offA,$ldA,$XX,$offX,$incX);
    }

    public function testMaximumMinusN()
    {
        $matlib = $this->getMatlib();

        $A = $this->array([[1,2],[2,3],[3,4]]);
        $X = $this->array([2,3]);
        [$M,$N,$AA,$offA,$ldA,$XX,$offX,$incX] =
            $this->translate_maximum($A,$X);

        $N = 0;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument n must be greater than 0.');
        $matlib->maximum($M,$N,$AA,$offA,$ldA,$XX,$offX,$incX);
    }

    public function testMaximumMinusOffsetA()
    {
        $matlib = $this->getMatlib();

        $A = $this->array([[1,2],[2,3],[3,4]]);
        $X = $this->array([2,3]);
        [$M,$N,$AA,$offA,$ldA,$XX,$offX,$incX] =
            $this->translate_maximum($A,$X);

        $offA = -1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument offsetA must be greater than or equals 0.');
        $matlib->maximum($M,$N,$AA,$offA,$ldA,$XX,$offX,$incX);
    }

    public function testMaximumMinusLdA()
    {
        $matlib = $this->getMatlib();

        $A = $this->array([[1,2],[2,3],[3,4]]);
        $X = $this->array([2,3]);
        [$M,$N,$AA,$offA,$ldA,$XX,$offX,$incX] =
            $this->translate_maximum($A,$X);

        $ldA = -1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument ldA must be greater than 0.');
        $matlib->maximum($M,$N,$AA,$offA,$ldA,$XX,$offX,$incX);
    }

    public function testMaximumIllegalBufferA()
    {
        $matlib = $this->getMatlib();

        $A = $this->array([[1,2],[2,3],[3,4]]);
        $X = $this->array([2,3]);
        [$M,$N,$AA,$offA,$ldA,$XX,$offX,$incX] =
            $this->translate_maximum($A,$X);

        $AA = new \stdClass();
        $this->expectException(TypeError::class);
        $this->expectExceptionMessage('must be of type Interop\Polite\Math\Matrix\LinearBuffer');
        $matlib->maximum($M,$N,$AA,$offA,$ldA,$XX,$offX,$incX);
    }

    public function testMaximumOverflowBufferAwithSize()
    {
        $matlib = $this->getMatlib();

        $A = $this->array([[1,2],[2,3],[3,4]]);
        $X = $this->array([2,3]);
        [$M,$N,$AA,$offA,$ldA,$XX,$offX,$incX] =
            $this->translate_maximum($A,$X);

        $AA = $this->array([1,2,2,3,3])->buffer();
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Matrix specification too large for bufferA');
        $matlib->maximum($M,$N,$AA,$offA,$ldA,$XX,$offX,$incX);
    }

    public function testMaximumOverflowBufferXwithOffsetA()
    {
        $matlib = $this->getMatlib();

        $A = $this->array([[1,2],[2,3],[3,4]]);
        $X = $this->array([2,3]);
        [$M,$N,$AA,$offA,$ldA,$XX,$offX,$incX] =
            $this->translate_maximum($A,$X);

        $offA = 1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Matrix specification too large for bufferA');
        $matlib->maximum($M,$N,$AA,$offA,$ldA,$XX,$offX,$incX);
    }

    public function testMaximumOverflowBufferXwithLdA()
    {
        $matlib = $this->getMatlib();

        $A = $this->array([[1,2],[2,3],[3,4]]);
        $X = $this->array([2,3]);
        [$M,$N,$AA,$offA,$ldA,$XX,$offX,$incX] =
            $this->translate_maximum($A,$X);

        $ldA = 3;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Matrix specification too large for bufferA');
        $matlib->maximum($M,$N,$AA,$offA,$ldA,$XX,$offX,$incX);
    }

    public function testMaximumMinusOffsetX()
    {
        $matlib = $this->getMatlib();

        $A = $this->array([[1,2],[2,3],[3,4]]);
        $X = $this->array([2,3]);
        [$M,$N,$AA,$offA,$ldA,$XX,$offX,$incX] =
            $this->translate_maximum($A,$X);

        $offX = -1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument offsetX must be greater than or equals 0.');
        $matlib->maximum($M,$N,$AA,$offA,$ldA,$XX,$offX,$incX);
    }

    public function testMaximumMinusIncX()
    {
        $matlib = $this->getMatlib();

        $A = $this->array([[1,2],[2,3],[3,4]]);
        $X = $this->array([2,3]);
        [$M,$N,$AA,$offA,$ldA,$XX,$offX,$incX] =
            $this->translate_maximum($A,$X);

        $incX = -1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument incX must be greater than 0.');
        $matlib->maximum($M,$N,$AA,$offA,$ldA,$XX,$offX,$incX);
    }

    public function testMaximumIllegalBufferX()
    {
        $matlib = $this->getMatlib();

        $A = $this->array([[1,2],[2,3],[3,4]]);
        $X = $this->array([2,3]);
        [$M,$N,$AA,$offA,$ldA,$XX,$offX,$incX] =
            $this->translate_maximum($A,$X);

        $XX = new \stdClass();
        $this->expectException(TypeError::class);
        $this->expectExceptionMessage('must be of type Interop\Polite\Math\Matrix\LinearBuffer');
        $matlib->maximum($M,$N,$AA,$offA,$ldA,$XX,$offX,$incX);
    }

    public function testMaximumOverflowBufferXwithSize()
    {
        $matlib = $this->getMatlib();

        $A = $this->array([[1,2],[2,3],[3,4]]);
        $X = $this->array([2,3]);
        [$M,$N,$AA,$offA,$ldA,$XX,$offX,$incX] =
            $this->translate_maximum($A,$X);

        $XX = $this->array([1])->buffer();
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Vector specification too large for bufferX');
        $matlib->maximum($M,$N,$AA,$offA,$ldA,$XX,$offX,$incX);
    }

    public function testMaximumOverflowBufferXwithOffsetX()
    {
        $matlib = $this->getMatlib();

        $A = $this->array([[1,2],[2,3],[3,4]]);
        $X = $this->array([2,3]);
        [$M,$N,$AA,$offA,$ldA,$XX,$offX,$incX] =
            $this->translate_maximum($A,$X);

        $offX = 1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Vector specification too large for bufferX');
        $matlib->maximum($M,$N,$AA,$offA,$ldA,$XX,$offX,$incX);
    }

    public function testMaximumOverflowBufferXwithIncX()
    {
        $matlib = $this->getMatlib();

        $A = $this->array([[1,2],[2,3],[3,4]]);
        $X = $this->array([2,3]);
        [$M,$N,$AA,$offA,$ldA,$XX,$offX,$incX] =
            $this->translate_maximum($A,$X);

        $incX = 2;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Vector specification too large for bufferX');
        $matlib->maximum($M,$N,$AA,$offA,$ldA,$XX,$offX,$incX);
    }

    #[DataProvider('providerDtypesFloats')]
    public function testMinimumNormal($params)
    {
        extract($params);
        $matlib = $this->getMatlib();

        $A = $this->array([[1,2],[2,3],[3,4]],dtype:$dtype);
        $X = $this->array([2,3],dtype:$dtype);
        [$M,$N,$AA,$offA,$ldA,$XX,$offX,$incX] =
            $this->translate_maximum($A,$X);

        $matlib->minimum($M,$N,$AA,$offA,$ldA,$XX,$offX,$incX);
        $this->assertEquals([[1,2],[2,3],[2,3]],$A->toArray());
    }

    public function testMinimumMinusM()
    {
        $matlib = $this->getMatlib();

        $A = $this->array([[1,2],[2,3],[3,4]]);
        $X = $this->array([2,3]);
        [$M,$N,$AA,$offA,$ldA,$XX,$offX,$incX] =
            $this->translate_maximum($A,$X);

        $M = 0;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument m must be greater than 0.');
        $matlib->minimum($M,$N,$AA,$offA,$ldA,$XX,$offX,$incX);
    }

    public function testMinimumMinusN()
    {
        $matlib = $this->getMatlib();

        $A = $this->array([[1,2],[2,3],[3,4]]);
        $X = $this->array([2,3]);
        [$M,$N,$AA,$offA,$ldA,$XX,$offX,$incX] =
            $this->translate_maximum($A,$X);

        $N = 0;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument n must be greater than 0.');
        $matlib->minimum($M,$N,$AA,$offA,$ldA,$XX,$offX,$incX);
    }

    public function testMinimumMinusOffsetA()
    {
        $matlib = $this->getMatlib();

        $A = $this->array([[1,2],[2,3],[3,4]]);
        $X = $this->array([2,3]);
        [$M,$N,$AA,$offA,$ldA,$XX,$offX,$incX] =
            $this->translate_maximum($A,$X);

        $offA = -1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument offsetA must be greater than or equals 0.');
        $matlib->minimum($M,$N,$AA,$offA,$ldA,$XX,$offX,$incX);
    }

    public function testMinimumMinusLdA()
    {
        $matlib = $this->getMatlib();

        $A = $this->array([[1,2],[2,3],[3,4]]);
        $X = $this->array([2,3]);
        [$M,$N,$AA,$offA,$ldA,$XX,$offX,$incX] =
            $this->translate_maximum($A,$X);

        $ldA = -1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument ldA must be greater than 0.');
        $matlib->minimum($M,$N,$AA,$offA,$ldA,$XX,$offX,$incX);
    }

    public function testMinimumIllegalBufferA()
    {
        $matlib = $this->getMatlib();

        $A = $this->array([[1,2],[2,3],[3,4]]);
        $X = $this->array([2,3]);
        [$M,$N,$AA,$offA,$ldA,$XX,$offX,$incX] =
            $this->translate_maximum($A,$X);

        $AA = new \stdClass();
        $this->expectException(TypeError::class);
        $this->expectExceptionMessage('must be of type Interop\Polite\Math\Matrix\LinearBuffer');
        $matlib->minimum($M,$N,$AA,$offA,$ldA,$XX,$offX,$incX);
    }

    public function testMinimumOverflowBufferAwithSize()
    {
        $matlib = $this->getMatlib();

        $A = $this->array([[1,2],[2,3],[3,4]]);
        $X = $this->array([2,3]);
        [$M,$N,$AA,$offA,$ldA,$XX,$offX,$incX] =
            $this->translate_maximum($A,$X);

        $AA = $this->array([1,2,2,3,3])->buffer();
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Matrix specification too large for bufferA');
        $matlib->minimum($M,$N,$AA,$offA,$ldA,$XX,$offX,$incX);
    }

    public function testMinimumOverflowBufferXwithOffsetA()
    {
        $matlib = $this->getMatlib();

        $A = $this->array([[1,2],[2,3],[3,4]]);
        $X = $this->array([2,3]);
        [$M,$N,$AA,$offA,$ldA,$XX,$offX,$incX] =
            $this->translate_maximum($A,$X);

        $offA = 1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Matrix specification too large for bufferA');
        $matlib->minimum($M,$N,$AA,$offA,$ldA,$XX,$offX,$incX);
    }

    public function testMinimumOverflowBufferXwithLdA()
    {
        $matlib = $this->getMatlib();

        $A = $this->array([[1,2],[2,3],[3,4]]);
        $X = $this->array([2,3]);
        [$M,$N,$AA,$offA,$ldA,$XX,$offX,$incX] =
            $this->translate_maximum($A,$X);

        $ldA = 3;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Matrix specification too large for bufferA');
        $matlib->minimum($M,$N,$AA,$offA,$ldA,$XX,$offX,$incX);
    }

    public function testMinimumMinusOffsetX()
    {
        $matlib = $this->getMatlib();

        $A = $this->array([[1,2],[2,3],[3,4]]);
        $X = $this->array([2,3]);
        [$M,$N,$AA,$offA,$ldA,$XX,$offX,$incX] =
            $this->translate_maximum($A,$X);

        $offX = -1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument offsetX must be greater than or equals 0.');
        $matlib->minimum($M,$N,$AA,$offA,$ldA,$XX,$offX,$incX);
    }

    public function testMinimumMinusIncX()
    {
        $matlib = $this->getMatlib();

        $A = $this->array([[1,2],[2,3],[3,4]]);
        $X = $this->array([2,3]);
        [$M,$N,$AA,$offA,$ldA,$XX,$offX,$incX] =
            $this->translate_maximum($A,$X);

        $incX = -1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument incX must be greater than 0.');
        $matlib->minimum($M,$N,$AA,$offA,$ldA,$XX,$offX,$incX);
    }

    public function testMinimumIllegalBufferX()
    {
        $matlib = $this->getMatlib();

        $A = $this->array([[1,2],[2,3],[3,4]]);
        $X = $this->array([2,3]);
        [$M,$N,$AA,$offA,$ldA,$XX,$offX,$incX] =
            $this->translate_maximum($A,$X);

        $XX = new \stdClass();
        $this->expectException(TypeError::class);
        $this->expectExceptionMessage('must be of type Interop\Polite\Math\Matrix\LinearBuffer');
        $matlib->minimum($M,$N,$AA,$offA,$ldA,$XX,$offX,$incX);
    }

    public function testMinimumOverflowBufferXwithSize()
    {
        $matlib = $this->getMatlib();

        $A = $this->array([[1,2],[2,3],[3,4]]);
        $X = $this->array([2,3]);
        [$M,$N,$AA,$offA,$ldA,$XX,$offX,$incX] =
            $this->translate_maximum($A,$X);

        $XX = $this->array([1])->buffer();
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Vector specification too large for bufferX');
        $matlib->minimum($M,$N,$AA,$offA,$ldA,$XX,$offX,$incX);
    }

    public function testMinimumOverflowBufferXwithOffsetX()
    {
        $matlib = $this->getMatlib();

        $A = $this->array([[1,2],[2,3],[3,4]]);
        $X = $this->array([2,3]);
        [$M,$N,$AA,$offA,$ldA,$XX,$offX,$incX] =
            $this->translate_maximum($A,$X);

        $offX = 1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Vector specification too large for bufferX');
        $matlib->minimum($M,$N,$AA,$offA,$ldA,$XX,$offX,$incX);
    }

    public function testMinimumOverflowBufferXwithIncX()
    {
        $matlib = $this->getMatlib();

        $A = $this->array([[1,2],[2,3],[3,4]]);
        $X = $this->array([2,3]);
        [$M,$N,$AA,$offA,$ldA,$XX,$offX,$incX] =
            $this->translate_maximum($A,$X);

        $incX = 2;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Vector specification too large for bufferX');
        $matlib->minimum($M,$N,$AA,$offA,$ldA,$XX,$offX,$incX);
    }
    
    #[DataProvider('providerDtypesFloats')]
    public function testGreaterNormal($params)
    {
        extract($params);
        $matlib = $this->getMatlib();

        $A = $this->array([[1,2],[2,3],[3,4]],dtype:$dtype);
        $X = $this->array([2,3],dtype:$dtype);
        [$M,$N,$AA,$offA,$ldA,$XX,$offX,$incX] =
            $this->translate_maximum($A,$X);

        $matlib->greater($M,$N,$AA,$offA,$ldA,$XX,$offX,$incX);
        $this->assertEquals([[0,0],[0,0],[1,1]],$A->toArray());
    }

    public function testGreaterMinusM()
    {
        $matlib = $this->getMatlib();

        $A = $this->array([[1,2],[2,3],[3,4]]);
        $X = $this->array([2,3]);
        [$M,$N,$AA,$offA,$ldA,$XX,$offX,$incX] =
            $this->translate_maximum($A,$X);

        $M = 0;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument m must be greater than 0.');
        $matlib->greater($M,$N,$AA,$offA,$ldA,$XX,$offX,$incX);
    }

    public function testGreaterMinusN()
    {
        $matlib = $this->getMatlib();

        $A = $this->array([[1,2],[2,3],[3,4]]);
        $X = $this->array([2,3]);
        [$M,$N,$AA,$offA,$ldA,$XX,$offX,$incX] =
            $this->translate_maximum($A,$X);

        $N = 0;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument n must be greater than 0.');
        $matlib->greater($M,$N,$AA,$offA,$ldA,$XX,$offX,$incX);
    }

    public function testGreaterMinusOffsetA()
    {
        $matlib = $this->getMatlib();

        $A = $this->array([[1,2],[2,3],[3,4]]);
        $X = $this->array([2,3]);
        [$M,$N,$AA,$offA,$ldA,$XX,$offX,$incX] =
            $this->translate_maximum($A,$X);

        $offA = -1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument offsetA must be greater than or equals 0.');
        $matlib->greater($M,$N,$AA,$offA,$ldA,$XX,$offX,$incX);
    }

    public function testGreaterMinusLdA()
    {
        $matlib = $this->getMatlib();

        $A = $this->array([[1,2],[2,3],[3,4]]);
        $X = $this->array([2,3]);
        [$M,$N,$AA,$offA,$ldA,$XX,$offX,$incX] =
            $this->translate_maximum($A,$X);

        $ldA = -1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument ldA must be greater than 0.');
        $matlib->greater($M,$N,$AA,$offA,$ldA,$XX,$offX,$incX);
    }

    public function testGreaterIllegalBufferA()
    {
        $matlib = $this->getMatlib();

        $A = $this->array([[1,2],[2,3],[3,4]]);
        $X = $this->array([2,3]);
        [$M,$N,$AA,$offA,$ldA,$XX,$offX,$incX] =
            $this->translate_maximum($A,$X);

        $AA = new \stdClass();
        $this->expectException(TypeError::class);
        $this->expectExceptionMessage('must be of type Interop\Polite\Math\Matrix\LinearBuffer');
        $matlib->greater($M,$N,$AA,$offA,$ldA,$XX,$offX,$incX);
    }

    public function testGreaterOverflowBufferAwithSize()
    {
        $matlib = $this->getMatlib();

        $A = $this->array([[1,2],[2,3],[3,4]]);
        $X = $this->array([2,3]);
        [$M,$N,$AA,$offA,$ldA,$XX,$offX,$incX] =
            $this->translate_maximum($A,$X);

        $AA = $this->array([1,2,2,3,3])->buffer();
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Matrix specification too large for bufferA');
        $matlib->greater($M,$N,$AA,$offA,$ldA,$XX,$offX,$incX);
    }

    public function testGreaterOverflowBufferXwithOffsetA()
    {
        $matlib = $this->getMatlib();

        $A = $this->array([[1,2],[2,3],[3,4]]);
        $X = $this->array([2,3]);
        [$M,$N,$AA,$offA,$ldA,$XX,$offX,$incX] =
            $this->translate_maximum($A,$X);

        $offA = 1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Matrix specification too large for bufferA');
        $matlib->greater($M,$N,$AA,$offA,$ldA,$XX,$offX,$incX);
    }

    public function testGreaterOverflowBufferXwithLdA()
    {
        $matlib = $this->getMatlib();

        $A = $this->array([[1,2],[2,3],[3,4]]);
        $X = $this->array([2,3]);
        [$M,$N,$AA,$offA,$ldA,$XX,$offX,$incX] =
            $this->translate_maximum($A,$X);

        $ldA = 3;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Matrix specification too large for bufferA');
        $matlib->greater($M,$N,$AA,$offA,$ldA,$XX,$offX,$incX);
    }

    public function testGreaterMinusOffsetX()
    {
        $matlib = $this->getMatlib();

        $A = $this->array([[1,2],[2,3],[3,4]]);
        $X = $this->array([2,3]);
        [$M,$N,$AA,$offA,$ldA,$XX,$offX,$incX] =
            $this->translate_maximum($A,$X);

        $offX = -1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument offsetX must be greater than or equals 0.');
        $matlib->greater($M,$N,$AA,$offA,$ldA,$XX,$offX,$incX);
    }

    public function testGreaterMinusIncX()
    {
        $matlib = $this->getMatlib();

        $A = $this->array([[1,2],[2,3],[3,4]]);
        $X = $this->array([2,3]);
        [$M,$N,$AA,$offA,$ldA,$XX,$offX,$incX] =
            $this->translate_maximum($A,$X);

        $incX = -1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument incX must be greater than 0.');
        $matlib->greater($M,$N,$AA,$offA,$ldA,$XX,$offX,$incX);
    }

    public function testGreaterIllegalBufferX()
    {
        $matlib = $this->getMatlib();

        $A = $this->array([[1,2],[2,3],[3,4]]);
        $X = $this->array([2,3]);
        [$M,$N,$AA,$offA,$ldA,$XX,$offX,$incX] =
            $this->translate_maximum($A,$X);

        $XX = new \stdClass();
        $this->expectException(TypeError::class);
        $this->expectExceptionMessage('must be of type Interop\Polite\Math\Matrix\LinearBuffer');
        $matlib->greater($M,$N,$AA,$offA,$ldA,$XX,$offX,$incX);
    }

    public function testGreaterOverflowBufferXwithSize()
    {
        $matlib = $this->getMatlib();

        $A = $this->array([[1,2],[2,3],[3,4]]);
        $X = $this->array([2,3]);
        [$M,$N,$AA,$offA,$ldA,$XX,$offX,$incX] =
            $this->translate_maximum($A,$X);

        $XX = $this->array([1])->buffer();
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Vector specification too large for bufferX');
        $matlib->greater($M,$N,$AA,$offA,$ldA,$XX,$offX,$incX);
    }

    public function testGreaterOverflowBufferXwithOffsetX()
    {
        $matlib = $this->getMatlib();

        $A = $this->array([[1,2],[2,3],[3,4]]);
        $X = $this->array([2,3]);
        [$M,$N,$AA,$offA,$ldA,$XX,$offX,$incX] =
            $this->translate_maximum($A,$X);

        $offX = 1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Vector specification too large for bufferX');
        $matlib->greater($M,$N,$AA,$offA,$ldA,$XX,$offX,$incX);
    }

    public function testGreaterOverflowBufferXwithIncX()
    {
        $matlib = $this->getMatlib();

        $A = $this->array([[1,2],[2,3],[3,4]]);
        $X = $this->array([2,3]);
        [$M,$N,$AA,$offA,$ldA,$XX,$offX,$incX] =
            $this->translate_maximum($A,$X);

        $incX = 2;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Vector specification too large for bufferX');
        $matlib->greater($M,$N,$AA,$offA,$ldA,$XX,$offX,$incX);
    }

    #[DataProvider('providerDtypesFloats')]
    public function testGreaterEqualNormal($params)
    {
        extract($params);
        $matlib = $this->getMatlib();

        $A = $this->array([[1,2],[2,3],[3,4]],dtype:$dtype);
        $X = $this->array([2,3],dtype:$dtype);
        [$M,$N,$AA,$offA,$ldA,$XX,$offX,$incX] =
            $this->translate_maximum($A,$X);

        $matlib->greaterEqual($M,$N,$AA,$offA,$ldA,$XX,$offX,$incX);
        $this->assertEquals([[0,0],[1,1],[1,1]],$A->toArray());
    }

    #[DataProvider('providerDtypesFloats')]
    public function testLessNormal($params)
    {
        extract($params);
        $matlib = $this->getMatlib();

        $A = $this->array([[1,2],[2,3],[3,4]],dtype:$dtype);
        $X = $this->array([2,3],dtype:$dtype);
        [$M,$N,$AA,$offA,$ldA,$XX,$offX,$incX] =
            $this->translate_maximum($A,$X);

        $matlib->less($M,$N,$AA,$offA,$ldA,$XX,$offX,$incX);
        $this->assertEquals([[1,1],[0,0],[0,0]],$A->toArray());
    }

    public function testLessMinusM()
    {
        $matlib = $this->getMatlib();

        $A = $this->array([[1,2],[2,3],[3,4]]);
        $X = $this->array([2,3]);
        [$M,$N,$AA,$offA,$ldA,$XX,$offX,$incX] =
            $this->translate_maximum($A,$X);

        $M = 0;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument m must be greater than 0.');
        $matlib->less($M,$N,$AA,$offA,$ldA,$XX,$offX,$incX);
    }

    public function testLessMinusN()
    {
        $matlib = $this->getMatlib();

        $A = $this->array([[1,2],[2,3],[3,4]]);
        $X = $this->array([2,3]);
        [$M,$N,$AA,$offA,$ldA,$XX,$offX,$incX] =
            $this->translate_maximum($A,$X);

        $N = 0;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument n must be greater than 0.');
        $matlib->less($M,$N,$AA,$offA,$ldA,$XX,$offX,$incX);
    }

    public function testLessMinusOffsetA()
    {
        $matlib = $this->getMatlib();

        $A = $this->array([[1,2],[2,3],[3,4]]);
        $X = $this->array([2,3]);
        [$M,$N,$AA,$offA,$ldA,$XX,$offX,$incX] =
            $this->translate_maximum($A,$X);

        $offA = -1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument offsetA must be greater than or equals 0.');
        $matlib->less($M,$N,$AA,$offA,$ldA,$XX,$offX,$incX);
    }

    public function testLessMinusLdA()
    {
        $matlib = $this->getMatlib();

        $A = $this->array([[1,2],[2,3],[3,4]]);
        $X = $this->array([2,3]);
        [$M,$N,$AA,$offA,$ldA,$XX,$offX,$incX] =
            $this->translate_maximum($A,$X);

        $ldA = -1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument ldA must be greater than 0.');
        $matlib->less($M,$N,$AA,$offA,$ldA,$XX,$offX,$incX);
    }

    public function testLessIllegalBufferA()
    {
        $matlib = $this->getMatlib();

        $A = $this->array([[1,2],[2,3],[3,4]]);
        $X = $this->array([2,3]);
        [$M,$N,$AA,$offA,$ldA,$XX,$offX,$incX] =
            $this->translate_maximum($A,$X);

        $AA = new \stdClass();
        $this->expectException(TypeError::class);
        $this->expectExceptionMessage('must be of type Interop\Polite\Math\Matrix\LinearBuffer');
        $matlib->less($M,$N,$AA,$offA,$ldA,$XX,$offX,$incX);
    }

    public function testLessOverflowBufferAwithSize()
    {
        $matlib = $this->getMatlib();

        $A = $this->array([[1,2],[2,3],[3,4]]);
        $X = $this->array([2,3]);
        [$M,$N,$AA,$offA,$ldA,$XX,$offX,$incX] =
            $this->translate_maximum($A,$X);

        $AA = $this->array([1,2,2,3,3])->buffer();
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Matrix specification too large for bufferA');
        $matlib->less($M,$N,$AA,$offA,$ldA,$XX,$offX,$incX);
    }

    public function testLessOverflowBufferXwithOffsetA()
    {
        $matlib = $this->getMatlib();

        $A = $this->array([[1,2],[2,3],[3,4]]);
        $X = $this->array([2,3]);
        [$M,$N,$AA,$offA,$ldA,$XX,$offX,$incX] =
            $this->translate_maximum($A,$X);

        $offA = 1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Matrix specification too large for bufferA');
        $matlib->less($M,$N,$AA,$offA,$ldA,$XX,$offX,$incX);
    }

    public function testLessOverflowBufferXwithLdA()
    {
        $matlib = $this->getMatlib();

        $A = $this->array([[1,2],[2,3],[3,4]]);
        $X = $this->array([2,3]);
        [$M,$N,$AA,$offA,$ldA,$XX,$offX,$incX] =
            $this->translate_maximum($A,$X);

        $ldA = 3;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Matrix specification too large for bufferA');
        $matlib->less($M,$N,$AA,$offA,$ldA,$XX,$offX,$incX);
    }

    public function testLessMinusOffsetX()
    {
        $matlib = $this->getMatlib();

        $A = $this->array([[1,2],[2,3],[3,4]]);
        $X = $this->array([2,3]);
        [$M,$N,$AA,$offA,$ldA,$XX,$offX,$incX] =
            $this->translate_maximum($A,$X);

        $offX = -1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument offsetX must be greater than or equals 0.');
        $matlib->less($M,$N,$AA,$offA,$ldA,$XX,$offX,$incX);
    }

    public function testLessMinusIncX()
    {
        $matlib = $this->getMatlib();

        $A = $this->array([[1,2],[2,3],[3,4]]);
        $X = $this->array([2,3]);
        [$M,$N,$AA,$offA,$ldA,$XX,$offX,$incX] =
            $this->translate_maximum($A,$X);

        $incX = -1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument incX must be greater than 0.');
        $matlib->less($M,$N,$AA,$offA,$ldA,$XX,$offX,$incX);
    }

    public function testLessIllegalBufferX()
    {
        $matlib = $this->getMatlib();

        $A = $this->array([[1,2],[2,3],[3,4]]);
        $X = $this->array([2,3]);
        [$M,$N,$AA,$offA,$ldA,$XX,$offX,$incX] =
            $this->translate_maximum($A,$X);

        $XX = new \stdClass();
        $this->expectException(TypeError::class);
        $this->expectExceptionMessage('must be of type Interop\Polite\Math\Matrix\LinearBuffer');
        $matlib->less($M,$N,$AA,$offA,$ldA,$XX,$offX,$incX);
    }

    public function testLessOverflowBufferXwithSize()
    {
        $matlib = $this->getMatlib();

        $A = $this->array([[1,2],[2,3],[3,4]]);
        $X = $this->array([2,3]);
        [$M,$N,$AA,$offA,$ldA,$XX,$offX,$incX] =
            $this->translate_maximum($A,$X);

        $XX = $this->array([1])->buffer();
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Vector specification too large for bufferX');
        $matlib->less($M,$N,$AA,$offA,$ldA,$XX,$offX,$incX);
    }

    public function testLessOverflowBufferXwithOffsetX()
    {
        $matlib = $this->getMatlib();

        $A = $this->array([[1,2],[2,3],[3,4]]);
        $X = $this->array([2,3]);
        [$M,$N,$AA,$offA,$ldA,$XX,$offX,$incX] =
            $this->translate_maximum($A,$X);

        $offX = 1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Vector specification too large for bufferX');
        $matlib->less($M,$N,$AA,$offA,$ldA,$XX,$offX,$incX);
    }

    public function testLessOverflowBufferXwithIncX()
    {
        $matlib = $this->getMatlib();

        $A = $this->array([[1,2],[2,3],[3,4]]);
        $X = $this->array([2,3]);
        [$M,$N,$AA,$offA,$ldA,$XX,$offX,$incX] =
            $this->translate_maximum($A,$X);

        $incX = 2;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Vector specification too large for bufferX');
        $matlib->less($M,$N,$AA,$offA,$ldA,$XX,$offX,$incX);
    }

    #[DataProvider('providerDtypesFloats')]
    public function testLessEqualNormal($params)
    {
        extract($params);
        $matlib = $this->getMatlib();

        $A = $this->array([[1,2],[2,3],[3,4]],dtype:$dtype);
        $X = $this->array([2,3],dtype:$dtype);
        [$M,$N,$AA,$offA,$ldA,$XX,$offX,$incX] =
            $this->translate_maximum($A,$X);

        $matlib->lessEqual($M,$N,$AA,$offA,$ldA,$XX,$offX,$incX);
        $this->assertEquals([[1,1],[1,1],[0,0]],$A->toArray());
    }

    #[DataProvider('providerDtypesFloats')]
    public function testMultiplySameSizeNormal($params)
    {
        extract($params);
        if($this->checkSkip('multiply')){return;}

        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3],dtype:$dtype);
        $A = $this->array([10,100,1000],dtype:$dtype);
        [$trans,$M,$N,$XX,$offX,$incX,$AA,$offA,$ldA] =
            $this->translate_multiply($X,$A);

        $matlib->multiply($trans,$M,$N,$XX,$offX,$incX,$AA,$offA,$ldA);
        $this->assertEquals([10,200,3000],$A->toArray());
    }

    #[DataProvider('providerDtypesFloats')]
    public function testMultiplyBroadcastNormal($params)
    {
        extract($params);
        if($this->checkSkip('multiply')){return;}

        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3],dtype:$dtype);
        $A = $this->array([[10,100,1000],[-1,-1,-1]],dtype:$dtype);
        [$trans,$M,$N,$XX,$offX,$incX,$AA,$offA,$ldA] =
            $this->translate_multiply($X,$A);

        $matlib->multiply($trans,$M,$N,$XX,$offX,$incX,$AA,$offA,$ldA);
        $this->assertEquals([[10,200,3000],[-1,-2,-3]],$A->toArray());
    }

    public function testMultiplyBroadcastTranspose()
    {
        if($this->checkSkip('multiply')){return;}

        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        $A = $this->array([[10,100],[1000,10000],[-1,-1]]);
        [$trans,$M,$N,$XX,$offX,$incX,$AA,$offA,$ldA] =
            $this->translate_multiply($X,$A,true);

        $matlib->multiply($trans,$M,$N,$XX,$offX,$incX,$AA,$offA,$ldA);
        $this->assertEquals([[10,100],[2000,20000],[-3,-3]],$A->toArray());
    }

    public function testMultiplyMinusM()
    {
        if($this->checkSkip('multiply')){return;}

        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        $A = $this->array([10,100,1000]);
        [$trans,$M,$N,$XX,$offX,$incX,$AA,$offA,$ldA] =
            $this->translate_multiply($X,$A);

        $M = 0;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument m must be greater than 0.');
        $matlib->multiply($trans,$M,$N,$XX,$offX,$incX,$AA,$offA,$ldA);
    }

    public function testMultiplyMinusN()
    {
        if($this->checkSkip('multiply')){return;}

        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        $A = $this->array([10,100,1000]);
        [$trans,$M,$N,$XX,$offX,$incX,$AA,$offA,$ldA] =
            $this->translate_multiply($X,$A);

        $N = 0;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument n must be greater than 0.');
        $matlib->multiply($trans,$M,$N,$XX,$offX,$incX,$AA,$offA,$ldA);
    }

    public function testMultiplyMinusOffsetX()
    {
        if($this->checkSkip('multiply')){return;}

        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        $A = $this->array([10,100,1000]);
        [$trans,$M,$N,$XX,$offX,$incX,$AA,$offA,$ldA] =
            $this->translate_multiply($X,$A);

        $offX = -1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument offsetX must be greater than or equals 0.');
        $matlib->multiply($trans,$M,$N,$XX,$offX,$incX,$AA,$offA,$ldA);
    }

    public function testMultiplyMinusIncX()
    {
        if($this->checkSkip('multiply')){return;}

        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        $A = $this->array([10,100,1000]);
        [$trans,$M,$N,$XX,$offX,$incX,$AA,$offA,$ldA] =
            $this->translate_multiply($X,$A);

        $incX = 0;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument incX must be greater than 0.');
        $matlib->multiply($trans,$M,$N,$XX,$offX,$incX,$AA,$offA,$ldA);
    }

    public function testMultiplyIllegalBufferX()
    {
        if($this->checkSkip('multiply')){return;}

        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        $A = $this->array([10,100,1000]);
        [$trans,$M,$N,$XX,$offX,$incX,$AA,$offA,$ldA] =
            $this->translate_multiply($X,$A);

        $XX = new \stdClass();
        $this->expectException(TypeError::class);
        $this->expectExceptionMessage('must be of type Interop\Polite\Math\Matrix\LinearBuffer');
        $matlib->multiply($trans,$M,$N,$XX,$offX,$incX,$AA,$offA,$ldA);
    }

    public function testMultiplyOverflowBufferXwithSize()
    {
        if($this->checkSkip('multiply')){return;}

        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        $A = $this->array([10,100,1000]);
        [$trans,$M,$N,$XX,$offX,$incX,$AA,$offA,$ldA] =
            $this->translate_multiply($X,$A);

        $XX = $this->array([1,2])->buffer();
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Vector specification too large for bufferX');
        $matlib->multiply($trans,$M,$N,$XX,$offX,$incX,$AA,$offA,$ldA);
    }

    public function testMultiplyOverflowBufferXwithOffsetX()
    {
        if($this->checkSkip('multiply')){return;}

        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        $A = $this->array([10,100,1000]);
        [$trans,$M,$N,$XX,$offX,$incX,$AA,$offA,$ldA] =
            $this->translate_multiply($X,$A);

        $offX = 1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Vector specification too large for bufferX');
        $matlib->multiply($trans,$M,$N,$XX,$offX,$incX,$AA,$offA,$ldA);
    }

    public function testMultiplyOverflowBufferXwithIncX()
    {
        if($this->checkSkip('multiply')){return;}

        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        $A = $this->array([10,100,1000]);
        [$trans,$M,$N,$XX,$offX,$incX,$AA,$offA,$ldA] =
            $this->translate_multiply($X,$A);

        $incX = 2;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Vector specification too large for bufferX');
        $matlib->multiply($trans,$M,$N,$XX,$offX,$incX,$AA,$offA,$ldA);
    }

    public function testMultiplyMinusOffsetA()
    {
        if($this->checkSkip('multiply')){return;}

        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        $A = $this->array([10,100,1000]);
        [$trans,$M,$N,$XX,$offX,$incX,$AA,$offA,$ldA] =
            $this->translate_multiply($X,$A);

        $offA = -1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument offsetA must be greater than or equals 0.');
        $matlib->multiply($trans,$M,$N,$XX,$offX,$incX,$AA,$offA,$ldA);
    }

    public function testMultiplyMinusIncA()
    {
        if($this->checkSkip('multiply')){return;}

        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        $A = $this->array([10,100,1000]);
        [$trans,$M,$N,$XX,$offX,$incX,$AA,$offA,$ldA] =
            $this->translate_multiply($X,$A);

        $ldA = 0;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument ldA must be greater than 0.');
        $matlib->multiply($trans,$M,$N,$XX,$offX,$incX,$AA,$offA,$ldA);
    }

    public function testMultiplyIllegalBufferA()
    {
        if($this->checkSkip('multiply')){return;}

        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        $A = $this->array([10,100,1000]);
        [$trans,$M,$N,$XX,$offX,$incX,$AA,$offA,$ldA] =
            $this->translate_multiply($X,$A);

        $AA = new \stdClass();
        $this->expectException(TypeError::class);
        $this->expectExceptionMessage('must be of type Interop\Polite\Math\Matrix\LinearBuffer');
        $matlib->multiply($trans,$M,$N,$XX,$offX,$incX,$AA,$offA,$ldA);
    }

    public function testMultiplyOverflowBufferAwithSize()
    {
        if($this->checkSkip('multiply')){return;}

        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        $A = $this->array([10,100,1000]);
        [$trans,$M,$N,$XX,$offX,$incX,$AA,$offA,$ldA] =
            $this->translate_multiply($X,$A);

        $AA = $this->array([1,2])->buffer();
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Matrix specification too large for bufferA');
        $matlib->multiply($trans,$M,$N,$XX,$offX,$incX,$AA,$offA,$ldA);
    }

    public function testMultiplyOverflowBufferXwithOffsetA()
    {
        if($this->checkSkip('multiply')){return;}

        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        $A = $this->array([10,100,1000]);
        [$trans,$M,$N,$XX,$offX,$incX,$AA,$offA,$ldA] =
            $this->translate_multiply($X,$A);

        $offA = 1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Matrix specification too large for bufferA');
        $matlib->multiply($trans,$M,$N,$XX,$offX,$incX,$AA,$offA,$ldA);
    }

    public function testMultiplyOverflowBufferXwithLdA()
    {
        if($this->checkSkip('multiply')){return;}

        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        $A = $this->array([[10,100,1000],[10,100,1000]]);
        [$trans,$M,$N,$XX,$offX,$incX,$AA,$offA,$ldA] =
            $this->translate_multiply($X,$A);

        $ldA = 4;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Matrix specification too large for bufferA');
        $matlib->multiply($trans,$M,$N,$XX,$offX,$incX,$AA,$offA,$ldA);
    }

    #[DataProvider('providerDtypesFloats')]
    public function testaddSameSizeNormal($params)
    {
        extract($params);
        if($this->checkSkip('add')){return;}

        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3],dtype:$dtype);
        $A = $this->array([10,100,1000],dtype:$dtype);
        [$trans,$M,$N,$alpha,$XX,$offX,$incX,$AA,$offA,$ldA] =
            $this->translate_add($X,$A,-1);

        $this->assertEquals(1,$M);
        $this->assertEquals(3,$N);
        $matlib->add($trans,$M,$N,$alpha,$XX,$offX,$incX,$AA,$offA,$ldA);
        $this->assertEquals([9,98,997],$A->toArray());
    }

    #[DataProvider('providerDtypesFloats')]
    public function testaddBroadcastNormal($params)
    {
        extract($params);
        if($this->checkSkip('add')){return;}

        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3],dtype:$dtype);
        $A = $this->array([[10,100,1000],[-1,-1,-1]],dtype:$dtype);
        [$trans,$M,$N,$alpha,$XX,$offX,$incX,$AA,$offA,$ldA] =
            $this->translate_add($X,$A);

        $matlib->add($trans,$M,$N,$alpha,$XX,$offX,$incX,$AA,$offA,$ldA);
        $this->assertEquals([[11,102,1003],[0,1,2]],$A->toArray());
    }

    public function testaddBroadcastTranspose()
    {
        if($this->checkSkip('add')){return;}

        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        $A = $this->array([[10,100],[1000,10000],[-1,-1]]);
        [$trans,$M,$N,$alpha,$XX,$offX,$incX,$AA,$offA,$ldA] =
            $this->translate_add($X,$A,null,true);

        $matlib->add($trans,$M,$N,$alpha,$XX,$offX,$incX,$AA,$offA,$ldA);
        $this->assertEquals([[11,101],[1002,10002],[2,2]],$A->toArray());
    }

    public function testaddMinusM()
    {
        if($this->checkSkip('add')){return;}

        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        $A = $this->array([10,100,1000]);
        [$trans,$M,$N,$alpha,$XX,$offX,$incX,$AA,$offA,$ldA] =
            $this->translate_add($X,$A);

        $M = 0;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument m must be greater than 0.');
        $matlib->add($trans,$M,$N,$alpha,$XX,$offX,$incX,$AA,$offA,$ldA);
    }

    public function testaddMinusN()
    {
        if($this->checkSkip('add')){return;}

        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        $A = $this->array([10,100,1000]);
        [$trans,$M,$N,$alpha,$XX,$offX,$incX,$AA,$offA,$ldA] =
            $this->translate_add($X,$A);

        $N = 0;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument n must be greater than 0.');
        $matlib->add($trans,$M,$N,$alpha,$XX,$offX,$incX,$AA,$offA,$ldA);
    }

    public function testaddMinusOffsetX()
    {
        if($this->checkSkip('add')){return;}

        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        $A = $this->array([10,100,1000]);
        [$trans,$M,$N,$alpha,$XX,$offX,$incX,$AA,$offA,$ldA] =
            $this->translate_add($X,$A);

        $offX = -1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument offsetX must be greater than or equals 0.');
        $matlib->add($trans,$M,$N,$alpha,$XX,$offX,$incX,$AA,$offA,$ldA);
    }

    public function testaddMinusIncX()
    {
        if($this->checkSkip('add')){return;}

        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        $A = $this->array([10,100,1000]);
        [$trans,$M,$N,$alpha,$XX,$offX,$incX,$AA,$offA,$ldA] =
            $this->translate_add($X,$A);

        $incX = 0;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument incX must be greater than 0.');
        $matlib->add($trans,$M,$N,$alpha,$XX,$offX,$incX,$AA,$offA,$ldA);
    }

    public function testaddIllegalBufferX()
    {
        if($this->checkSkip('add')){return;}

        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        $A = $this->array([10,100,1000]);
        [$trans,$M,$N,$alpha,$XX,$offX,$incX,$AA,$offA,$ldA] =
            $this->translate_add($X,$A);

        $XX = new \stdClass();
        $this->expectException(TypeError::class);
        $this->expectExceptionMessage('must be of type Interop\Polite\Math\Matrix\LinearBuffer');
        $matlib->add($trans,$M,$N,$alpha,$XX,$offX,$incX,$AA,$offA,$ldA);
    }

    public function testaddOverflowBufferXwithSize()
    {
        if($this->checkSkip('add')){return;}

        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        $A = $this->array([10,100,1000]);
        [$trans,$M,$N,$alpha,$XX,$offX,$incX,$AA,$offA,$ldA] =
            $this->translate_add($X,$A);

        $XX = $this->array([1,2])->buffer();
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Vector specification too large for bufferX');
        $matlib->add($trans,$M,$N,$alpha,$XX,$offX,$incX,$AA,$offA,$ldA);
    }

    public function testaddOverflowBufferXwithOffsetX()
    {
        if($this->checkSkip('add')){return;}

        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        $A = $this->array([10,100,1000]);
        [$trans,$M,$N,$alpha,$XX,$offX,$incX,$AA,$offA,$ldA] =
            $this->translate_add($X,$A);

        $offX = 1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Vector specification too large for bufferX');
        $matlib->add($trans,$M,$N,$alpha,$XX,$offX,$incX,$AA,$offA,$ldA);
    }

    public function testaddOverflowBufferXwithIncX()
    {
        if($this->checkSkip('add')){return;}

        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        $A = $this->array([10,100,1000]);
        [$trans,$M,$N,$alpha,$XX,$offX,$incX,$AA,$offA,$ldA] =
            $this->translate_add($X,$A);

        $incX = 2;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Vector specification too large for bufferX');
        $matlib->add($trans,$M,$N,$alpha,$XX,$offX,$incX,$AA,$offA,$ldA);
    }

    public function testaddMinusOffsetA()
    {
        if($this->checkSkip('add')){return;}

        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        $A = $this->array([10,100,1000]);
        [$trans,$M,$N,$alpha,$XX,$offX,$incX,$AA,$offA,$ldA] =
            $this->translate_add($X,$A);

        $offA = -1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument offsetA must be greater than or equals 0.');
        $matlib->add($trans,$M,$N,$alpha,$XX,$offX,$incX,$AA,$offA,$ldA);
    }

    public function testaddMinusIncA()
    {
        if($this->checkSkip('add')){return;}

        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        $A = $this->array([10,100,1000]);
        [$trans,$M,$N,$alpha,$XX,$offX,$incX,$AA,$offA,$ldA] =
            $this->translate_add($X,$A);

        $ldA = 0;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument ldA must be greater than 0.');
        $matlib->add($trans,$M,$N,$alpha,$XX,$offX,$incX,$AA,$offA,$ldA);
    }

    public function testaddIllegalBufferA()
    {
        if($this->checkSkip('add')){return;}

        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        $A = $this->array([10,100,1000]);
        [$trans,$M,$N,$alpha,$XX,$offX,$incX,$AA,$offA,$ldA] =
            $this->translate_add($X,$A);

        $AA = new \stdClass();
        $this->expectException(TypeError::class);
        $this->expectExceptionMessage('must be of type Interop\Polite\Math\Matrix\LinearBuffer');
        $matlib->add($trans,$M,$N,$alpha,$XX,$offX,$incX,$AA,$offA,$ldA);
    }

    public function testaddOverflowBufferAwithSize()
    {
        if($this->checkSkip('add')){return;}

        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        $A = $this->array([10,100,1000]);
        [$trans,$M,$N,$alpha,$XX,$offX,$incX,$AA,$offA,$ldA] =
            $this->translate_add($X,$A);

        $AA = $this->array([1,2])->buffer();
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Matrix specification too large for bufferA');
        $matlib->add($trans,$M,$N,$alpha,$XX,$offX,$incX,$AA,$offA,$ldA);
    }

    public function testaddOverflowBufferXwithOffsetA()
    {
        if($this->checkSkip('add')){return;}

        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        $A = $this->array([10,100,1000]);
        [$trans,$M,$N,$alpha,$XX,$offX,$incX,$AA,$offA,$ldA] =
            $this->translate_add($X,$A);

        $offA = 1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Matrix specification too large for bufferA');
        $matlib->add($trans,$M,$N,$alpha,$XX,$offX,$incX,$AA,$offA,$ldA);
    }

    public function testaddOverflowBufferXwithLdA()
    {
        if($this->checkSkip('add')){return;}

        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        $A = $this->array([[10,100,1000],[10,100,1000]]);
        [$trans,$M,$N,$alpha,$XX,$offX,$incX,$AA,$offA,$ldA] =
            $this->translate_add($X,$A);

        $ldA = 4;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Matrix specification too large for bufferA');
        $matlib->add($trans,$M,$N,$alpha,$XX,$offX,$incX,$AA,$offA,$ldA);
    }

    #[DataProvider('providerDtypesFloats')]
    public function testDuplicateSameSizeNormal($params)
    {
        extract($params);
        if($this->checkSkip('duplicate')){return;}

        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3],dtype:$dtype);
        $A = $this->array([10,100,1000],dtype:$dtype);
        [$trans,$M,$N,$XX,$offX,$incX,$AA,$offA,$ldA] =
            $this->translate_duplicate($X,null,null,$A);

        $matlib->duplicate($trans,$M,$N,$XX,$offX,$incX,$AA,$offA,$ldA);
        $this->assertEquals([1,2,3],$A->toArray());
    }

    #[DataProvider('providerDtypesFloats')]
    public function testDuplicateBroadcastNormal($params)
    {
        extract($params);
        if($this->checkSkip('duplicate')){return;}

        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3],dtype:$dtype);
        $A = $this->array([[10,100,1000],[-1,-1,-1]],dtype:$dtype);
        [$trans,$M,$N,$XX,$offX,$incX,$AA,$offA,$ldA] =
            $this->translate_duplicate($X,null,null,$A);

        $matlib->duplicate($trans,$M,$N,$XX,$offX,$incX,$AA,$offA,$ldA);
        $this->assertEquals([[1,2,3],[1,2,3]],$A->toArray());
    }

    public function testDuplicateBroadcastTranspose()
    {
        if($this->checkSkip('duplicate')){return;}

        $matlib = $this->getMatlib();

        $X = $this->array([1,2]);
        $A = $this->array([[10,100,1000],[-1,-1,-1]]);
        [$trans,$M,$N,$XX,$offX,$incX,$AA,$offA,$ldA] =
            $this->translate_duplicate($X,null,true,$A);
    
        $matlib->duplicate($trans,$M,$N,$XX,$offX,$incX,$AA,$offA,$ldA);
        $this->assertEquals([[1,1,1],[2,2,2]],$A->toArray());
    }

    public function testDuplicateMinusM()
    {
        if($this->checkSkip('duplicate')){return;}

        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        $A = $this->array([10,100,1000]);
        [$trans,$M,$N,$XX,$offX,$incX,$AA,$offA,$ldA] =
            $this->translate_duplicate($X,null,null,$A);

        $M = 0;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument m must be greater than 0.');
        $matlib->duplicate($trans,$M,$N,$XX,$offX,$incX,$AA,$offA,$ldA);
    }

    public function testDuplicateMinusN()
    {
        if($this->checkSkip('duplicate')){return;}

        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        $A = $this->array([10,100,1000]);
        [$trans,$M,$N,$XX,$offX,$incX,$AA,$offA,$ldA] =
            $this->translate_duplicate($X,null,null,$A);

        $N = 0;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument n must be greater than 0.');
        $matlib->duplicate($trans,$M,$N,$XX,$offX,$incX,$AA,$offA,$ldA);
    }

    public function testDuplicateMinusOffsetX()
    {
        if($this->checkSkip('duplicate')){return;}

        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        $A = $this->array([10,100,1000]);
        [$trans,$M,$N,$XX,$offX,$incX,$AA,$offA,$ldA] =
            $this->translate_duplicate($X,null,null,$A);

        $offX = -1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument offsetX must be greater than or equals 0.');
        $matlib->duplicate($trans,$M,$N,$XX,$offX,$incX,$AA,$offA,$ldA);
    }

    public function testDuplicateMinusIncX()
    {
        if($this->checkSkip('duplicate')){return;}

        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        $A = $this->array([10,100,1000]);
        [$trans,$M,$N,$XX,$offX,$incX,$AA,$offA,$ldA] =
            $this->translate_duplicate($X,null,null,$A);

        $incX = 0;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument incX must be greater than 0.');
        $matlib->duplicate($trans,$M,$N,$XX,$offX,$incX,$AA,$offA,$ldA);
    }

    public function testDuplicateIllegalBufferX()
    {
        if($this->checkSkip('duplicate')){return;}

        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        $A = $this->array([10,100,1000]);
        [$trans,$M,$N,$XX,$offX,$incX,$AA,$offA,$ldA] =
            $this->translate_duplicate($X,null,null,$A);

        $XX = new \stdClass();
        $this->expectException(TypeError::class);
        $this->expectExceptionMessage('must be of type Interop\Polite\Math\Matrix\LinearBuffer');
        $matlib->duplicate($trans,$M,$N,$XX,$offX,$incX,$AA,$offA,$ldA);
    }

    public function testDuplicateOverflowBufferXwithSize()
    {
        if($this->checkSkip('duplicate')){return;}

        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        $A = $this->array([10,100,1000]);
        [$trans,$M,$N,$XX,$offX,$incX,$AA,$offA,$ldA] =
            $this->translate_duplicate($X,null,null,$A);

        $XX = $this->array([2])->buffer();
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Vector specification too large for bufferX');
        $matlib->duplicate($trans,$M,$N,$XX,$offX,$incX,$AA,$offA,$ldA);
    }

    public function testDuplicateOverflowBufferXwithOffsetX()
    {
        if($this->checkSkip('duplicate')){return;}

        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        $A = $this->array([10,100,1000]);
        [$trans,$M,$N,$XX,$offX,$incX,$AA,$offA,$ldA] =
            $this->translate_duplicate($X,null,null,$A);

        $offX = 1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Vector specification too large for bufferX');
        $matlib->duplicate($trans,$M,$N,$XX,$offX,$incX,$AA,$offA,$ldA);
    }

    public function testDuplicateOverflowBufferXwithIncX()
    {
        if($this->checkSkip('duplicate')){return;}

        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        $A = $this->array([10,100,1000]);
        [$trans,$M,$N,$XX,$offX,$incX,$AA,$offA,$ldA] =
            $this->translate_duplicate($X,null,null,$A);

        $incX = 2;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Vector specification too large for bufferX');
        $matlib->duplicate($trans,$M,$N,$XX,$offX,$incX,$AA,$offA,$ldA);
    }

    public function testDuplicateMinusOffsetA()
    {
        if($this->checkSkip('duplicate')){return;}

        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        $A = $this->array([10,100,1000]);
        [$trans,$M,$N,$XX,$offX,$incX,$AA,$offA,$ldA] =
            $this->translate_duplicate($X,null,null,$A);

        $offA = -1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument offsetA must be greater than or equals 0.');
        $matlib->duplicate($trans,$M,$N,$XX,$offX,$incX,$AA,$offA,$ldA);
    }

    public function testDuplicateMinusLdA()
    {
        if($this->checkSkip('duplicate')){return;}

        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        $A = $this->array([10,100,1000]);
        [$trans,$M,$N,$XX,$offX,$incX,$AA,$offA,$ldA] =
            $this->translate_duplicate($X,null,null,$A);

        $ldA = 0;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument ldA must be greater than 0.');
        $matlib->duplicate($trans,$M,$N,$XX,$offX,$incX,$AA,$offA,$ldA);
    }

    public function testDuplicateIllegalBufferA()
    {
        if($this->checkSkip('duplicate')){return;}

        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        $A = $this->array([10,100,1000]);
        [$trans,$M,$N,$XX,$offX,$incX,$AA,$offA,$ldA] =
            $this->translate_duplicate($X,null,null,$A);

        $AA = new \stdClass();
        $this->expectException(TypeError::class);
        $this->expectExceptionMessage('must be of type Interop\Polite\Math\Matrix\LinearBuffer');
        $matlib->duplicate($trans,$M,$N,$XX,$offX,$incX,$AA,$offA,$ldA);
    }

    public function testDuplicateOverflowBufferAwithSize()
    {
        if($this->checkSkip('duplicate')){return;}

        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        $A = $this->array([10,100,1000]);
        [$trans,$M,$N,$XX,$offX,$incX,$AA,$offA,$ldA] =
            $this->translate_duplicate($X,null,null,$A);

        $AA = $this->array([1,2])->buffer();
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Matrix specification too large for bufferA');
        $matlib->duplicate($trans,$M,$N,$XX,$offX,$incX,$AA,$offA,$ldA);
    }

    public function testDuplicateOverflowBufferAwithOffsetA()
    {
        if($this->checkSkip('duplicate')){return;}

        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        $A = $this->array([10,100,1000]);
        [$trans,$M,$N,$XX,$offX,$incX,$AA,$offA,$ldA] =
            $this->translate_duplicate($X,null,null,$A);

        $offA = 1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Matrix specification too large for bufferA');
        $matlib->duplicate($trans,$M,$N,$XX,$offX,$incX,$AA,$offA,$ldA);
    }

    public function testDuplicateOverflowBufferAwithLdA()
    {
        if($this->checkSkip('duplicate')){return;}

        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        $A = $this->array([[10,100,1000],[10,100,1000]]);
        [$trans,$M,$N,$XX,$offX,$incX,$AA,$offA,$ldA] =
            $this->translate_duplicate($X,null,null,$A);

        $ldA = 4;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Matrix specification too large for bufferA');
        $matlib->duplicate($trans,$M,$N,$XX,$offX,$incX,$AA,$offA,$ldA);
    }

    #[DataProvider('providerDtypesFloats')]
    public function testSquareNormal($params)
    {
        extract($params);
        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3],dtype:$dtype);
        [$N,$XX,$offX,$incX] =
            $this->translate_square($X);

        $matlib->square($N,$XX,$offX,$incX);
        $this->assertEquals([1,4,9],$X->toArray());
    }

    public function testsquareMinusN()
    {
        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        [$N,$XX,$offX,$incX] =
            $this->translate_square($X);

        $N = 0;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument n must be greater than 0.');
        $matlib->square($N,$XX,$offX,$incX);
    }

    public function testsquareMinusOffsetX()
    {
        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        [$N,$XX,$offX,$incX] =
            $this->translate_square($X);

        $offX = -1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument offsetX must be greater than or equals 0.');
        $matlib->square($N,$XX,$offX,$incX);
    }

    public function testsquareMinusIncX()
    {
        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        [$N,$XX,$offX,$incX] =
            $this->translate_square($X);

        $incX = 0;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument incX must be greater than 0.');
        $matlib->square($N,$XX,$offX,$incX);
    }

    public function testsquareIllegalBufferX()
    {
        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        [$N,$XX,$offX,$incX] =
            $this->translate_square($X);

        $XX = new \stdClass();
        $this->expectException(TypeError::class);
        $this->expectExceptionMessage('must be of type Interop\Polite\Math\Matrix\LinearBuffer');
        $matlib->square($N,$XX,$offX,$incX);
    }

    public function testsquareOverflowBufferXwithSize()
    {
        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        [$N,$XX,$offX,$incX] =
            $this->translate_square($X);

        $XX = $this->array([1,2])->buffer();
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Vector specification too large for bufferX');
        $matlib->square($N,$XX,$offX,$incX);
    }

    public function testsquareOverflowBufferXwithOffsetX()
    {
        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        [$N,$XX,$offX,$incX] =
            $this->translate_square($X);

        $offX = 1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Vector specification too large for bufferX');
        $matlib->square($N,$XX,$offX,$incX);
    }

    public function testsquareOverflowBufferXwithIncX()
    {
        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        [$N,$XX,$offX,$incX] =
            $this->translate_square($X);

        $incX = 2;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Vector specification too large for bufferX');
        $matlib->square($N,$XX,$offX,$incX);
    }

    #[DataProvider('providerDtypesFloats')]
    public function testsqrtNormal($params)
    {
        extract($params);
        $matlib = $this->getMatlib();

        $X = $this->array([0,1,4,9],dtype:$dtype);
        [$N,$XX,$offX,$incX] =
            $this->translate_square($X);

        $matlib->sqrt($N,$XX,$offX,$incX);
        $this->assertEquals([0,1,2,3],$X->toArray());
    }

    public function testsqrtIllegalValue()
    {
        $matlib = $this->getMatlib();

        $X = $this->array([1,4,-1]);
        [$N,$XX,$offX,$incX] =
            $this->translate_square($X);

        // *** CAUTION ***
        // disable checking for INFINITY values
        //$this->expectException(RuntimeException::class);
        //$this->expectExceptionMessage('Invalid value in sqrt.');

        $matlib->sqrt($N,$XX,$offX,$incX);
        $this->assertEquals(1.0, $X[0]);
        $this->assertEquals(2.0, $X[1]);
        $this->assertTrue(is_nan($X[2])); // -NAN
    }

    public function testsqrtMinusN()
    {
        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        [$N,$XX,$offX,$incX] =
            $this->translate_square($X);

        $N = 0;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument n must be greater than 0.');
        $matlib->sqrt($N,$XX,$offX,$incX);
    }

    public function testsqrtMinusOffsetX()
    {
        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        [$N,$XX,$offX,$incX] =
            $this->translate_square($X);

        $offX = -1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument offsetX must be greater than or equals 0.');
        $matlib->sqrt($N,$XX,$offX,$incX);
    }

    public function testsqrtMinusIncX()
    {
        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        [$N,$XX,$offX,$incX] =
            $this->translate_square($X);

        $incX = 0;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument incX must be greater than 0.');
        $matlib->sqrt($N,$XX,$offX,$incX);
    }

    public function testssqrtIllegalBufferX()
    {
        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        [$N,$XX,$offX,$incX] =
            $this->translate_square($X);

        $XX = new \stdClass();
        $this->expectException(TypeError::class);
        $this->expectExceptionMessage('must be of type Interop\Polite\Math\Matrix\LinearBuffer');
        $matlib->sqrt($N,$XX,$offX,$incX);
    }

    public function testsqrtOverflowBufferXwithSize()
    {
        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        [$N,$XX,$offX,$incX] =
            $this->translate_square($X);

        $XX = $this->array([1,2])->buffer();
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Vector specification too large for bufferX');
        $matlib->sqrt($N,$XX,$offX,$incX);
    }

    public function testsqrtOverflowBufferXwithOffsetX()
    {
        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        [$N,$XX,$offX,$incX] =
            $this->translate_square($X);

        $offX = 1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Vector specification too large for bufferX');
        $matlib->sqrt($N,$XX,$offX,$incX);
    }

    public function testsqrtOverflowBufferXwithIncX()
    {
        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        [$N,$XX,$offX,$incX] =
            $this->translate_square($X);

        $incX = 2;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Vector specification too large for bufferX');
        $matlib->sqrt($N,$XX,$offX,$incX);
    }

    #[DataProvider('providerDtypesFloats')]
    public function testrsqrtNormal($params)
    {
        extract($params);
        $matlib = $this->getMatlib();

        $X = $this->array([1,4,16],dtype:$dtype);
        [$N,$alpha,$XX,$offX,$incX,$beta] =
            $this->translate_increment($X,1,2);

        $matlib->rsqrt($N,$alpha,$XX,$offX,$incX,$beta);
        // X := 1 / (a * sqrt(X) + b)

        $matlib->reciprocal($N,1.0,$XX,$offX,$incX,0.0);
        $matlib->increment($N,1.0,$XX,$offX,$incX,-1.0);
        $matlib->increment($N,0.5,$XX,$offX,$incX,0);
        $matlib->square($N,$XX,$offX,$incX);

        $this->assertEquals([1,4,16],$X->toArray());
    }

    public function testrsqrtZeroDivide()
    {
        $matlib = $this->getMatlib();

        $X = $this->array([4,1,0]);
        [$N,$alpha,$XX,$offX,$incX,$beta] =
            $this->translate_increment($X,$beta=0,$alpha=1);

        // X := 1 / (a * sqrt(X) + b)

        // *** CAUTION ***
        // disable checking for INFINITY values
        //$this->expectException(RuntimeException::class);
        //$this->expectExceptionMessage('Zero divide.');

        $matlib->rsqrt($N,$alpha,$XX,$offX,$incX,$beta);
        $this->assertEquals(
            [0.5, 1.0, INF],
            $X->toArray());
    }

    public function testrsqrtInvalidSqrt()
    {
        $matlib = $this->getMatlib();

        $X = $this->array([4,1,-1]);
        [$N,$alpha,$XX,$offX,$incX,$beta] =
            $this->translate_increment($X,0,1);

        // X := 1 / (a * sqrt(X) + b)
        // *** CAUTION ***
        // disable checking for INFINITY values
        //$this->expectException(RuntimeException::class);
        //$this->expectExceptionMessage('Invalid value in sqrt.');

        $matlib->rsqrt($N,$alpha,$XX,$offX,$incX,$beta);
        $this->assertEquals(0.5, $X[0]);
        $this->assertEquals(1.0, $X[1]);
        $this->assertTrue(is_nan($X[2]));
    }

    public function testrsqrtMinusN()
    {
        $matlib = $this->getMatlib();

        $X = $this->array([3,2,0]);
        [$N,$alpha,$XX,$offX,$incX,$beta] =
            $this->translate_increment($X,4,-1);

        $N = 0;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument n must be greater than 0.');
        $matlib->rsqrt($N,$alpha,$XX,$offX,$incX,$beta);
    }

    public function testrsqrtMinusOffsetX()
    {
        $matlib = $this->getMatlib();

        $X = $this->array([3,2,0]);
        [$N,$alpha,$XX,$offX,$incX,$beta] =
            $this->translate_increment($X,4,-1);

        $offX = -1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument offsetX must be greater than or equals 0.');
        $matlib->rsqrt($N,$alpha,$XX,$offX,$incX,$beta);
    }

    public function testrsqrtMinusIncX()
    {
        $matlib = $this->getMatlib();

        $X = $this->array([3,2,0]);
        [$N,$alpha,$XX,$offX,$incX,$beta] =
            $this->translate_increment($X,4,-1);

        $incX = 0;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument incX must be greater than 0.');
        $matlib->rsqrt($N,$alpha,$XX,$offX,$incX,$beta);
    }

    public function testrsqrtIllegalBufferX()
    {
        $matlib = $this->getMatlib();

        $X = $this->array([3,2,0]);
        [$N,$alpha,$XX,$offX,$incX,$beta] =
            $this->translate_increment($X,4,-1);

        $XX = new \stdClass();
        $this->expectException(TypeError::class);
        $this->expectExceptionMessage('must be of type Interop\Polite\Math\Matrix\LinearBuffer');
        $matlib->rsqrt($N,$alpha,$XX,$offX,$incX,$beta);
    }

    public function testrsqrtOverflowBufferXwithSize()
    {
        $matlib = $this->getMatlib();

        $X = $this->array([3,2,0]);
        [$N,$alpha,$XX,$offX,$incX,$beta] =
            $this->translate_increment($X,4,-1);

        $XX = $this->array([3,2])->buffer();
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Vector specification too large for bufferX');
        $matlib->rsqrt($N,$alpha,$XX,$offX,$incX,$beta);
    }

    public function testrsqrtOverflowBufferXwithOffsetX()
    {
        $matlib = $this->getMatlib();

        $X = $this->array([3,2,0]);
        [$N,$alpha,$XX,$offX,$incX,$beta] =
            $this->translate_increment($X,4,-1);

        $offX = 1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Vector specification too large for bufferX');
        $matlib->rsqrt($N,$alpha,$XX,$offX,$incX,$beta);
    }

    public function testrsqrtOverflowBufferXwithIncX()
    {
        $matlib = $this->getMatlib();

        $X = $this->array([3,2,0]);
        [$N,$alpha,$XX,$offX,$incX,$beta] =
            $this->translate_increment($X,4,-1);

        $incX = 2;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Vector specification too large for bufferX');
        $matlib->rsqrt($N,$alpha,$XX,$offX,$incX,$beta);
    }

    #[DataProvider('providerDtypesFloats')]
    public function testPowSameSizeNormal($params)
    {
        extract($params);
        if($this->checkSkip('pow')){return;}

        $matlib = $this->getMatlib();

        $A = $this->array([1,2,3],dtype:$dtype);
        $X = $this->array([4,3,2],dtype:$dtype);
        [$trans,$M,$N,$AA,$offA,$ldA,$XX,$offX,$incX] =
            $this->translate_pow($A,$X);

        $matlib->pow($trans,$M,$N,$AA,$offA,$ldA,$XX,$offX,$incX);;
        $this->assertEquals([1,8,9],$A->toArray());
    }

    #[DataProvider('providerDtypesFloats')]
    public function testPowBroadcastNormal($params)
    {
        extract($params);
        if($this->checkSkip('pow')){return;}

        $matlib = $this->getMatlib();

        $A = $this->array([[1,2,3],[4,5,6]]);
        $X = $this->array([4,3,2]);
        [$trans,$M,$N,$AA,$offA,$ldA,$XX,$offX,$incX] =
            $this->translate_pow($A,$X);

        $matlib->pow($trans,$M,$N,$AA,$offA,$ldA,$XX,$offX,$incX);;
        $this->assertEquals([[1,8,9],[256,125,36]],$A->toArray());
    }

    public function testPowBroadcastTranspose()
    {
        if($this->checkSkip('pow')){return;}

        $matlib = $this->getMatlib();

        $A = $this->array([[1,2,3],[4,5,6]]);
        $X = $this->array([3,2]);
        [$trans,$M,$N,$AA,$offA,$ldA,$XX,$offX,$incX] =
            $this->translate_pow($A,$X,trans:true);

        $matlib->pow($trans,$M,$N,$AA,$offA,$ldA,$XX,$offX,$incX);;
        $this->assertEquals([[1,8,27],[16,25,36]],$A->toArray());
    }

    public function testPowMinusM()
    {
        if($this->checkSkip('pow')){return;}

        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        $A = $this->array([10,100,1000]);
        [$trans,$M,$N,$AA,$offA,$ldA,$XX,$offX,$incX] =
            $this->translate_pow($A,$X);

        $M = 0;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument m must be greater than 0.');
        $matlib->pow($trans,$M,$N,$AA,$offA,$ldA,$XX,$offX,$incX);;
    }

    public function testPowMinusN()
    {
        if($this->checkSkip('pow')){return;}

        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        $A = $this->array([10,100,1000]);
        [$trans,$M,$N,$AA,$offA,$ldA,$XX,$offX,$incX] =
            $this->translate_pow($A,$X);

        $N = 0;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument n must be greater than 0.');
        $matlib->pow($trans,$M,$N,$AA,$offA,$ldA,$XX,$offX,$incX);;
    }

    public function testPowMinusOffsetX()
    {
        if($this->checkSkip('pow')){return;}

        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        $A = $this->array([10,100,1000]);
        [$trans,$M,$N,$AA,$offA,$ldA,$XX,$offX,$incX] =
            $this->translate_pow($A,$X);

        $offX = -1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument offsetX must be greater than or equals 0.');
        $matlib->pow($trans,$M,$N,$AA,$offA,$ldA,$XX,$offX,$incX);;
    }

    public function testPowMinusIncX()
    {
        if($this->checkSkip('pow')){return;}

        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        $A = $this->array([10,100,1000]);
        [$trans,$M,$N,$AA,$offA,$ldA,$XX,$offX,$incX] =
            $this->translate_pow($A,$X);

        $incX = 0;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument incX must be greater than 0.');
        $matlib->pow($trans,$M,$N,$AA,$offA,$ldA,$XX,$offX,$incX);;
    }

    public function testPowIllegalBufferX()
    {
        if($this->checkSkip('pow')){return;}

        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        $A = $this->array([10,100,1000]);
        [$trans,$M,$N,$AA,$offA,$ldA,$XX,$offX,$incX] =
            $this->translate_pow($A,$X);

        $XX = new \stdClass();
        $this->expectException(TypeError::class);
        $this->expectExceptionMessage('must be of type Interop\Polite\Math\Matrix\LinearBuffer');
        $matlib->pow($trans,$M,$N,$AA,$offA,$ldA,$XX,$offX,$incX);;
    }

    public function testPowOverflowBufferXwithSize()
    {
        if($this->checkSkip('pow')){return;}

        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        $A = $this->array([10,100,1000]);
        [$trans,$M,$N,$AA,$offA,$ldA,$XX,$offX,$incX] =
            $this->translate_pow($A,$X);

        $XX = $this->array([1,2])->buffer();
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Vector specification too large for bufferX');
        $matlib->pow($trans,$M,$N,$AA,$offA,$ldA,$XX,$offX,$incX);;
    }

    public function testPowOverflowBufferXwithOffsetX()
    {
        if($this->checkSkip('pow')){return;}

        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        $A = $this->array([10,100,1000]);
        [$trans,$M,$N,$AA,$offA,$ldA,$XX,$offX,$incX] =
            $this->translate_pow($A,$X);

        $offX = 1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Vector specification too large for bufferX');
        $matlib->pow($trans,$M,$N,$AA,$offA,$ldA,$XX,$offX,$incX);;
    }

    public function testPowOverflowBufferXwithIncX()
    {
        if($this->checkSkip('pow')){return;}

        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        $A = $this->array([10,100,1000]);
        [$trans,$M,$N,$AA,$offA,$ldA,$XX,$offX,$incX] =
            $this->translate_pow($A,$X);

        $incX = 2;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Vector specification too large for bufferX');
        $matlib->pow($trans,$M,$N,$AA,$offA,$ldA,$XX,$offX,$incX);;
    }

    public function testPowMinusOffsetA()
    {
        if($this->checkSkip('pow')){return;}

        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        $A = $this->array([10,100,1000]);
        [$trans,$M,$N,$AA,$offA,$ldA,$XX,$offX,$incX] =
            $this->translate_pow($A,$X);

        $offA = -1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument offsetA must be greater than or equals 0.');
        $matlib->pow($trans,$M,$N,$AA,$offA,$ldA,$XX,$offX,$incX);;
    }

    public function testPowMinusIncA()
    {
        if($this->checkSkip('pow')){return;}

        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        $A = $this->array([10,100,1000]);
        [$trans,$M,$N,$AA,$offA,$ldA,$XX,$offX,$incX] =
            $this->translate_pow($A,$X);

        $ldA = 0;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument ldA must be greater than 0.');
        $matlib->pow($trans,$M,$N,$AA,$offA,$ldA,$XX,$offX,$incX);;
    }

    public function testPowIllegalBufferA()
    {
        if($this->checkSkip('pow')){return;}

        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        $A = $this->array([10,100,1000]);
        [$trans,$M,$N,$AA,$offA,$ldA,$XX,$offX,$incX] =
            $this->translate_pow($A,$X);

        $AA = new \stdClass();
        $this->expectException(TypeError::class);
        $this->expectExceptionMessage('must be of type Interop\Polite\Math\Matrix\LinearBuffer');
        $matlib->pow($trans,$M,$N,$AA,$offA,$ldA,$XX,$offX,$incX);;
    }

    public function testPowOverflowBufferAwithSize()
    {
        if($this->checkSkip('pow')){return;}

        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        $A = $this->array([10,100,1000]);
        [$trans,$M,$N,$AA,$offA,$ldA,$XX,$offX,$incX] =
            $this->translate_pow($A,$X);

        $AA = $this->array([1,2])->buffer();
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Matrix specification too large for bufferA');
        $matlib->pow($trans,$M,$N,$AA,$offA,$ldA,$XX,$offX,$incX);;
    }

    public function testPowOverflowBufferXwithOffsetA()
    {
        if($this->checkSkip('pow')){return;}

        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        $A = $this->array([10,100,1000]);
        [$trans,$M,$N,$AA,$offA,$ldA,$XX,$offX,$incX] =
            $this->translate_pow($A,$X);

        $offA = 1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Matrix specification too large for bufferA');
        $matlib->pow($trans,$M,$N,$AA,$offA,$ldA,$XX,$offX,$incX);;
    }

    public function testPowOverflowBufferXwithLdA()
    {
        if($this->checkSkip('pow')){return;}

        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        $A = $this->array([[10,100,1000],[10,100,1000]]);
        [$trans,$M,$N,$AA,$offA,$ldA,$XX,$offX,$incX] =
            $this->translate_pow($A,$X);

        $ldA = 4;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Matrix specification too large for bufferA');
        $matlib->pow($trans,$M,$N,$AA,$offA,$ldA,$XX,$offX,$incX);;
    }

    #[DataProvider('providerDtypesFloats')]
    public function testexpNormal($params)
    {
        extract($params);
        $matlib = $this->getMatlib();

        $X = $this->array([0,2,4,9],dtype:$dtype);
        [$N,$XX,$offX,$incX] =
            $this->translate_square($X);

        $matlib->exp($N,$XX,$offX,$incX);
        $matlib->log($N,$XX,$offX,$incX);

        $this->assertEquals([0,2,4,9],$X->toArray());
    }

    public function testexpMinusN()
    {
        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        [$N,$XX,$offX,$incX] =
            $this->translate_square($X);

        $N = 0;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument n must be greater than 0.');
        $matlib->exp($N,$XX,$offX,$incX);
    }

    public function testexpMinusOffsetX()
    {
        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        [$N,$XX,$offX,$incX] =
            $this->translate_square($X);

        $offX = -1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument offsetX must be greater than or equals 0.');
        $matlib->exp($N,$XX,$offX,$incX);
    }

    public function testexpMinusIncX()
    {
        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        [$N,$XX,$offX,$incX] =
            $this->translate_square($X);

        $incX = 0;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument incX must be greater than 0.');
        $matlib->exp($N,$XX,$offX,$incX);
    }

    public function testexpIllegalBufferX()
    {
        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        [$N,$XX,$offX,$incX] =
            $this->translate_square($X);

        $XX = new \stdClass();
        $this->expectException(TypeError::class);
        $this->expectExceptionMessage('must be of type Interop\Polite\Math\Matrix\LinearBuffer');
        $matlib->exp($N,$XX,$offX,$incX);
    }

    public function testexpOverflowBufferXwithSize()
    {
        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        [$N,$XX,$offX,$incX] =
            $this->translate_square($X);

        $XX = $this->array([1,2])->buffer();
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Vector specification too large for bufferX');
        $matlib->exp($N,$XX,$offX,$incX);
    }

    public function testexpOverflowBufferXwithOffsetX()
    {
        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        [$N,$XX,$offX,$incX] =
            $this->translate_square($X);

        $offX = 1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Vector specification too large for bufferX');
        $matlib->exp($N,$XX,$offX,$incX);
    }

    public function testexpOverflowBufferXwithIncX()
    {
        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        [$N,$XX,$offX,$incX] =
            $this->translate_square($X);

        $incX = 2;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Vector specification too large for bufferX');
        $matlib->exp($N,$XX,$offX,$incX);
    }

    #[DataProvider('providerDtypesFloats')]
    public function testlogNormal($params)
    {
        extract($params);
        $matlib = $this->getMatlib();

        $X = $this->array([1,2,4,9],dtype:$dtype);
        [$N,$XX,$offX,$incX] =
            $this->translate_square($X);

        $matlib->log($N,$XX,$offX,$incX);
        $matlib->exp($N,$XX,$offX,$incX);

        $trues = $this->array([1,2,4,9],dtype:$dtype);
        $matlib->add(false,1,$N,-1,
            $XX,$offX,$incX,
            $trues->buffer(),$trues->offset(),$N,
        );
        $this->assertLessThan(1e-7,$matlib->sum($N,$trues->buffer(),$trues->offset(),1));
    }

    public function testlogInvalidValue()
    {
        $matlib = $this->getMatlib();

        $X = $this->array([1,0,-1]);
        [$N,$XX,$offX,$incX] =
            $this->translate_square($X);

        // *** CAUTION ***
        // disable checking for INFINITY values
        //$this->expectException(RuntimeException::class);
        //$this->expectExceptionMessage('Invalid value in log.');

        $matlib->log($N,$XX,$offX,$incX);
        $this->assertEquals(0.0,  $X[0]);
        $this->assertEquals(-INF, $X[1]);
        $this->assertTrue(is_nan($X[2]));
    }

    public function testlogMinusN()
    {
        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        [$N,$XX,$offX,$incX] =
            $this->translate_square($X);

        $N = 0;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument n must be greater than 0.');
        $matlib->log($N,$XX,$offX,$incX);
    }

    public function testlogMinusOffsetX()
    {
        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        [$N,$XX,$offX,$incX] =
            $this->translate_square($X);

        $offX = -1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument offsetX must be greater than or equals 0.');
        $matlib->log($N,$XX,$offX,$incX);
    }

    public function testlogMinusIncX()
    {
        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        [$N,$XX,$offX,$incX] =
            $this->translate_square($X);

        $incX = 0;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument incX must be greater than 0.');
        $matlib->log($N,$XX,$offX,$incX);
    }

    public function testlogIllegalBufferX()
    {
        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        [$N,$XX,$offX,$incX] =
            $this->translate_square($X);

        $XX = new \stdClass();
        $this->expectException(TypeError::class);
        $this->expectExceptionMessage('must be of type Interop\Polite\Math\Matrix\LinearBuffer');
        $matlib->log($N,$XX,$offX,$incX);
    }

    public function testlogOverflowBufferXwithSize()
    {
        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        [$N,$XX,$offX,$incX] =
            $this->translate_square($X);

        $XX = $this->array([1,2])->buffer();
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Vector specification too large for bufferX');
        $matlib->log($N,$XX,$offX,$incX);
    }

    public function testlogOverflowBufferXwithOffsetX()
    {
        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        [$N,$XX,$offX,$incX] =
            $this->translate_square($X);

        $offX = 1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Vector specification too large for bufferX');
        $matlib->log($N,$XX,$offX,$incX);
    }

    public function testlogOverflowBufferXwithIncX()
    {
        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        [$N,$XX,$offX,$incX] =
            $this->translate_square($X);

        $incX = 2;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Vector specification too large for bufferX');
        $matlib->log($N,$XX,$offX,$incX);
    }

    #[DataProvider('providerDtypesFloats')]
    public function testtanhNormal($params)
    {
        extract($params);
        $matlib = $this->getMatlib();

        $X = $this->array([1,2,4,9],dtype:$dtype);
        //$X = $this->array([1,0.9,0.7,0.5]);
        [$N,$XX,$offX,$incX] =
            $this->translate_square($X);

        $matlib->tanh($N,$XX,$offX,$incX);

        //$RS = $this->array([tanh(1),tanh(0.9),tanh(0.7),tanh(0.5)]);
        $RS = $this->array([tanh(1),tanh(2),tanh(4),tanh(9)],dtype:$dtype);
        $RR = $RS->buffer();
        $matlib->add($trans=false,$m=1,$N,$alpha=-1,$XX,$offX,$incX,$RR,$offRS=0,$incRS=1);
        $sum = $matlib->sum($N,$RR, $offRS, $incRS);
        $this->assertLessThan(0.1,$sum);
    }

    public function testtanhMinusN()
    {
        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        [$N,$XX,$offX,$incX] =
            $this->translate_square($X);

        $N = 0;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument n must be greater than 0.');
        $matlib->tanh($N,$XX,$offX,$incX);
    }

    public function testtanhMinusOffsetX()
    {
        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        [$N,$XX,$offX,$incX] =
            $this->translate_square($X);

        $offX = -1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument offsetX must be greater than or equals 0.');
        $matlib->tanh($N,$XX,$offX,$incX);
    }

    public function testtanhMinusIncX()
    {
        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        [$N,$XX,$offX,$incX] =
            $this->translate_square($X);

        $incX = 0;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument incX must be greater than 0.');
        $matlib->tanh($N,$XX,$offX,$incX);
    }

    public function testtanhIllegalBufferX()
    {
        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        [$N,$XX,$offX,$incX] =
            $this->translate_square($X);

        $XX = new \stdClass();
        $this->expectException(TypeError::class);
        $this->expectExceptionMessage('must be of type Interop\Polite\Math\Matrix\LinearBuffer');
        $matlib->tanh($N,$XX,$offX,$incX);
    }

    public function testtanhOverflowBufferXwithSize()
    {
        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        [$N,$XX,$offX,$incX] =
            $this->translate_square($X);

        $XX = $this->array([1,2])->buffer();
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Vector specification too large for bufferX');
        $matlib->tanh($N,$XX,$offX,$incX);
    }

    public function testtanhOverflowBufferXwithOffsetX()
    {
        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        [$N,$XX,$offX,$incX] =
            $this->translate_square($X);

        $offX = 1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Vector specification too large for bufferX');
        $matlib->tanh($N,$XX,$offX,$incX);
    }

    public function testtanhOverflowBufferXwithIncX()
    {
        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        [$N,$XX,$offX,$incX] =
            $this->translate_square($X);

        $incX = 2;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Vector specification too large for bufferX');
        $matlib->tanh($N,$XX,$offX,$incX);
    }

    #[DataProvider('providerDtypesFloats')]
    public function testsinNormal($params)
    {
        extract($params);
        $matlib = $this->getMatlib();

        $X = $this->array([1,2,4,9],dtype:$dtype);
        //$X = $this->array([1,0.9,0.7,0.5]);
        [$N,$XX,$offX,$incX] =
            $this->translate_square($X);

        $matlib->sin($N,$XX,$offX,$incX);

        //$RS = $this->array([sin(1),sin(0.9),sin(0.7),sin(0.5)]);
        $RS = $this->array([sin(1),sin(2),sin(4),sin(9)],dtype:$dtype);
        $RR = $RS->buffer();
        $matlib->add($trans=false,$m=1,$N,$alpha=-1,$XX,$offX,$incX,$RR,$offRS=0,$incRS=1);
        $sum = $matlib->sum($N,$RR, $offRS, $incRS);
        $this->assertLessThan(0.1,$sum);
    }

    public function testsinMinusN()
    {
        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        [$N,$XX,$offX,$incX] =
            $this->translate_square($X);

        $N = 0;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument n must be greater than 0.');
        $matlib->sin($N,$XX,$offX,$incX);
    }

    public function testsinMinusOffsetX()
    {
        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        [$N,$XX,$offX,$incX] =
            $this->translate_square($X);

        $offX = -1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument offsetX must be greater than or equals 0.');
        $matlib->sin($N,$XX,$offX,$incX);
    }

    public function testsinMinusIncX()
    {
        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        [$N,$XX,$offX,$incX] =
            $this->translate_square($X);

        $incX = 0;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument incX must be greater than 0.');
        $matlib->sin($N,$XX,$offX,$incX);
    }

    public function testsinIllegalBufferX()
    {
        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        [$N,$XX,$offX,$incX] =
            $this->translate_square($X);

        $XX = new \stdClass();
        $this->expectException(TypeError::class);
        $this->expectExceptionMessage('must be of type Interop\Polite\Math\Matrix\LinearBuffer');
        $matlib->sin($N,$XX,$offX,$incX);
    }

    public function testsinOverflowBufferXwithSize()
    {
        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        [$N,$XX,$offX,$incX] =
            $this->translate_square($X);

        $XX = $this->array([1,2])->buffer();
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Vector specification too large for bufferX');
        $matlib->sin($N,$XX,$offX,$incX);
    }

    public function testsinOverflowBufferXwithOffsetX()
    {
        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        [$N,$XX,$offX,$incX] =
            $this->translate_square($X);

        $offX = 1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Vector specification too large for bufferX');
        $matlib->sin($N,$XX,$offX,$incX);
    }

    #[DataProvider('providerDtypesFloats')]
    public function testcosNormal($params)
    {
        extract($params);
        $matlib = $this->getMatlib();

        $X = $this->array([1,2,4,9],dtype:$dtype);
        //$X = $this->array([1,0.9,0.7,0.5]);
        [$N,$XX,$offX,$incX] =
            $this->translate_square($X);

        $matlib->cos($N,$XX,$offX,$incX);

        //$RS = $this->array([cos(1),cos(0.9),cos(0.7),cos(0.5)]);
        $RS = $this->array([cos(1),cos(2),cos(4),cos(9)],dtype:$dtype);
        $RR = $RS->buffer();
        $matlib->add($trans=false,$m=1,$N,$alpha=-1,$XX,$offX,$incX,$RR,$offRS=0,$incRS=1);
        $sum = $matlib->sum($N,$RR, $offRS, $incRS);
        $this->assertLessThan(0.1,$sum);
    }

    public function testcosMinusN()
    {
        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        [$N,$XX,$offX,$incX] =
            $this->translate_square($X);

        $N = 0;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument n must be greater than 0.');
        $matlib->cos($N,$XX,$offX,$incX);
    }

    public function testcosMinusOffsetX()
    {
        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        [$N,$XX,$offX,$incX] =
            $this->translate_square($X);

        $offX = -1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument offsetX must be greater than or equals 0.');
        $matlib->cos($N,$XX,$offX,$incX);
    }

    public function testcosMinusIncX()
    {
        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        [$N,$XX,$offX,$incX] =
            $this->translate_square($X);

        $incX = 0;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument incX must be greater than 0.');
        $matlib->cos($N,$XX,$offX,$incX);
    }

    public function testcosIllegalBufferX()
    {
        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        [$N,$XX,$offX,$incX] =
            $this->translate_square($X);

        $XX = new \stdClass();
        $this->expectException(TypeError::class);
        $this->expectExceptionMessage('must be of type Interop\Polite\Math\Matrix\LinearBuffer');
        $matlib->cos($N,$XX,$offX,$incX);
    }

    public function testcosOverflowBufferXwithSize()
    {
        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        [$N,$XX,$offX,$incX] =
            $this->translate_square($X);

        $XX = $this->array([1,2])->buffer();
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Vector specification too large for bufferX');
        $matlib->cos($N,$XX,$offX,$incX);
    }

    public function testcosOverflowBufferXwithOffsetX()
    {
        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        [$N,$XX,$offX,$incX] =
            $this->translate_square($X);

        $offX = 1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Vector specification too large for bufferX');
        $matlib->cos($N,$XX,$offX,$incX);
    }

    public function testcosOverflowBufferXwithIncX()
    {
        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        [$N,$XX,$offX,$incX] =
            $this->translate_square($X);

        $incX = 2;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Vector specification too large for bufferX');
        $matlib->cos($N,$XX,$offX,$incX);
    }

    #[DataProvider('providerDtypesFloats')]
    public function testtanNormal($params)
    {
        extract($params);
        $matlib = $this->getMatlib();

        $X = $this->array([1,2,4,9],dtype:$dtype);
        //$X = $this->array([1,0.9,0.7,0.5]);
        [$N,$XX,$offX,$incX] =
            $this->translate_square($X);

        $matlib->tan($N,$XX,$offX,$incX);

        //$RS = $this->array([tan(1),tan(0.9),tan(0.7),tan(0.5)]);
        $RS = $this->array([tan(1),tan(2),tan(4),tan(9)],dtype:$dtype);
        $RR = $RS->buffer();
        $matlib->add($trans=false,$m=1,$N,$alpha=-1,$XX,$offX,$incX,$RR,$offRS=0,$incRS=1);
        $sum = $matlib->sum($N,$RR, $offRS, $incRS);
        $this->assertLessThan(0.1,$sum);
    }

    public function testtanMinusN()
    {
        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        [$N,$XX,$offX,$incX] =
            $this->translate_square($X);

        $N = 0;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument n must be greater than 0.');
        $matlib->tan($N,$XX,$offX,$incX);
    }

    public function testtanMinusOffsetX()
    {
        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        [$N,$XX,$offX,$incX] =
            $this->translate_square($X);

        $offX = -1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument offsetX must be greater than or equals 0.');
        $matlib->tan($N,$XX,$offX,$incX);
    }

    public function testtanMinusIncX()
    {
        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        [$N,$XX,$offX,$incX] =
            $this->translate_square($X);

        $incX = 0;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument incX must be greater than 0.');
        $matlib->tan($N,$XX,$offX,$incX);
    }

    public function testtanIllegalBufferX()
    {
        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        [$N,$XX,$offX,$incX] =
            $this->translate_square($X);

        $XX = new \stdClass();
        $this->expectException(TypeError::class);
        $this->expectExceptionMessage('must be of type Interop\Polite\Math\Matrix\LinearBuffer');
        $matlib->tan($N,$XX,$offX,$incX);
    }

    public function testtanOverflowBufferXwithSize()
    {
        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        [$N,$XX,$offX,$incX] =
            $this->translate_square($X);

        $XX = $this->array([1,2])->buffer();
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Vector specification too large for bufferX');
        $matlib->tan($N,$XX,$offX,$incX);
    }

    public function testtanOverflowBufferXwithOffsetX()
    {
        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        [$N,$XX,$offX,$incX] =
            $this->translate_square($X);

        $offX = 1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Vector specification too large for bufferX');
        $matlib->tan($N,$XX,$offX,$incX);
    }

    public function testtanOverflowBufferXwithIncX()
    {
        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        [$N,$XX,$offX,$incX] =
            $this->translate_square($X);

        $incX = 2;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Vector specification too large for bufferX');
        $matlib->tan($N,$XX,$offX,$incX);
    }

    #[DataProvider('providerDtypesFloats')]
    public function testfillNormal($params)
    {
        extract($params);
        $matlib = $this->getMatlib();

        $X = $this->array([NAN,NAN,NAN,NAN],dtype:$dtype);
        [$N, $VV, $offV, $XX, $offX, $incX] =
            $this->translate_fill($this->array(1.0,dtype:$X->dtype()),$X);
        $matlib->fill($N, $VV, $offV, $XX, $offX, $incX);
        $this->assertEquals([1,1,1,1],$X->toArray());
    }

    #[DataProvider('providerDtypesFloats')]
    public function testnan2numNormal($params)
    {
        extract($params);
        $matlib = $this->getMatlib();

        $X = $this->array([NAN,2,4,NAN],dtype:$dtype);
        [$N,$XX,$offX,$incX,$alpha] =
            $this->translate_nan2num($X,0.0);
        $matlib->nan2num($N,$XX,$offX,$incX,$alpha);
        $this->assertEquals([0,2,4,0],$X->toArray());

        $X = $this->array([NAN,2,4,NAN],dtype:$dtype);
        [$N,$XX,$offX,$incX,$alpha] =
            $this->translate_nan2num($X,1.0);
        $matlib->nan2num($N,$XX,$offX,$incX,$alpha);
        $this->assertEquals([1,2,4,1],$X->toArray());
    }

    #[DataProvider('providerDtypesFloats')]
    public function testisnanNormal($params)
    {
        extract($params);
        $matlib = $this->getMatlib();

        $X = $this->array([NAN,2,4,NAN],dtype:$dtype);
        [$N,$XX,$offX,$incX] =
            $this->translate_square($X);

        $matlib->isnan($N,$XX,$offX,$incX);

        $this->assertEquals([1,0,0,1],$X->toArray());
    }

    #[DataProvider('providerDtypesFloats')]
    public function testsearchsortedNormal($params)
    {
        extract($params);
        $matlib = $this->getMatlib();

        $A = $this->array([0,2,4],dtype:$dtype);
        $X = $this->array([-1,1,2,5],dtype:$dtype);
        $Y = $this->zeros([4],NDArray::int32);
        [$m,$n,$AA,$offsetA,$ldA,$XX,$offsetX,$incX,$right,$YY,$offsetY,$incY] =
            $this->translate_searchsorted($A,$X,false,null,$Y);

        $matlib->searchsorted($m,$n,$AA,$offsetA,$ldA,$XX,$offsetX,$incX,$right,$YY,$offsetY,$incY);
        $this->assertEquals([0,1,1,3],$Y->toArray());

        $matlib->searchsorted($m,$n,$AA,$offsetA,$ldA,$XX,$offsetX,$incX,true,$YY,$offsetY,$incY);
        $this->assertEquals([0,1,2,3],$Y->toArray());
    }

    public function testsearchsortedIndividual()
    {
        $matlib = $this->getMatlib();

        $A = $this->array([
            [1,   3,  5,   7,   9],
            [1,   2,  3,   4,   5],
            [0, 100, 20, 300, 400]
        ]);
        $X = $this->array([0, 5, 10]);
        $Y = $this->zeros([3],NDArray::int32);
        [$m,$n,$AA,$offsetA,$ldA,$XX,$offsetX,$incX,$right,$YY,$offsetY,$incY] =
            $this->translate_searchsorted($A,$X,false,null,$Y);

        $matlib->searchsorted($m,$n,$AA,$offsetA,$ldA,$XX,$offsetX,$incX,$right,$YY,$offsetY,$incY);
        $this->assertEquals([0, 4, 1],$Y->toArray());

        $matlib->searchsorted($m,$n,$AA,$offsetA,$ldA,$XX,$offsetX,$incX,true,$YY,$offsetY,$incY);
        $this->assertEquals([0, 5, 1],$Y->toArray());
    }

    public function testsearchsortedMinusM()
    {
        $matlib = $this->getMatlib();

        $A = $this->array([0,2,4]);
        $X = $this->array([-1,1,2,5]);
        $Y = $this->zeros([4],NDArray::int32);
        [$m,$n,$AA,$offsetA,$ldA,$XX,$offsetX,$incX,$right,$YY,$offsetY,$incY] =
            $this->translate_searchsorted($A,$X,false,null,$Y);

        $m = 0;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument m must be greater than 0.');
        $matlib->searchsorted($m,$n,$AA,$offsetA,$ldA,$XX,$offsetX,$incX,$right,$YY,$offsetY,$incY);
    }

    public function testsearchsortedMinusOffsetA()
    {
        $matlib = $this->getMatlib();

        $A = $this->array([0,2,4]);
        $X = $this->array([-1,1,2,5]);
        $Y = $this->zeros([4],NDArray::int32);
        [$m,$n,$AA,$offsetA,$ldA,$XX,$offsetX,$incX,$right,$YY,$offsetY,$incY] =
            $this->translate_searchsorted($A,$X,false,null,$Y);

        $offsetA = -1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument offsetA must be greater than or equals 0.');
        $matlib->searchsorted($m,$n,$AA,$offsetA,$ldA,$XX,$offsetX,$incX,$right,$YY,$offsetY,$incY);
    }

    public function testsearchsortedMinusldA()
    {
        $matlib = $this->getMatlib();

        $A = $this->array([0,2,4]);
        $X = $this->array([-1,1,2,5]);
        $Y = $this->zeros([4],NDArray::int32);
        [$m,$n,$AA,$offsetA,$ldA,$XX,$offsetX,$incX,$right,$YY,$offsetY,$incY] =
            $this->translate_searchsorted($A,$X,false,null,$Y);

        $ldA = -1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument ldA must be greater than or equals 0.');
        $matlib->searchsorted($m,$n,$AA,$offsetA,$ldA,$XX,$offsetX,$incX,$right,$YY,$offsetY,$incY);
    }

    public function testsearchsortedIllegalBufferA()
    {
        $matlib = $this->getMatlib();

        $A = $this->array([0,2,4]);
        $X = $this->array([-1,1,2,5]);
        $Y = $this->zeros([4],NDArray::int32);
        [$m,$n,$AA,$offsetA,$ldA,$XX,$offsetX,$incX,$right,$YY,$offsetY,$incY] =
            $this->translate_searchsorted($A,$X,false,null,$Y);

        $AA = new \stdClass();
        $this->expectException(TypeError::class);
        $this->expectExceptionMessage('must be of type Interop\Polite\Math\Matrix\LinearBuffer');
        $matlib->searchsorted($m,$n,$AA,$offsetA,$ldA,$XX,$offsetX,$incX,$right,$YY,$offsetY,$incY);
    }

    public function testsearchsortedOverflowBufferAwithSize()
    {
        $matlib = $this->getMatlib();

        $A = $this->array([0,2,4]);
        $X = $this->array([-1,1,2,5]);
        $Y = $this->zeros([4],NDArray::int32);
        [$m,$n,$AA,$offsetA,$ldA,$XX,$offsetX,$incX,$right,$YY,$offsetY,$incY] =
            $this->translate_searchsorted($A,$X,false,null,$Y);

        $AA = $this->array([1,2])->buffer();
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Matrix specification too large for bufferA');
        $matlib->searchsorted($m,$n,$AA,$offsetA,$ldA,$XX,$offsetX,$incX,$right,$YY,$offsetY,$incY);
    }

    public function testsearchsortedOverflowBufferAwithOffsetA()
    {
        $matlib = $this->getMatlib();

        $A = $this->array([0,2,4]);
        $X = $this->array([-1,1,2,5]);
        $Y = $this->zeros([4],NDArray::int32);
        [$m,$n,$AA,$offsetA,$ldA,$XX,$offsetX,$incX,$right,$YY,$offsetY,$incY] =
            $this->translate_searchsorted($A,$X,false,null,$Y);

        $offsetA = 1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Matrix specification too large for bufferA');
        $matlib->searchsorted($m,$n,$AA,$offsetA,$ldA,$XX,$offsetX,$incX,$right,$YY,$offsetY,$incY);
    }

    public function testsearchsortedOverflowBufferAwithIncA()
    {
        $matlib = $this->getMatlib();

        $A = $this->array([0,2,4]);
        $X = $this->array([-1,1,2,5]);
        $Y = $this->zeros([4],NDArray::int32);
        [$m,$n,$AA,$offsetA,$ldA,$XX,$offsetX,$incX,$right,$YY,$offsetY,$incY] =
            $this->translate_searchsorted($A,$X,false,null,$Y);

        $ldA = 2;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Matrix specification too large for bufferA');
        $matlib->searchsorted($m,$n,$AA,$offsetA,$ldA,$XX,$offsetX,$incX,$right,$YY,$offsetY,$incY);
    }

    public function testsearchsortedMinusN()
    {
        $matlib = $this->getMatlib();

        $A = $this->array([0,2,4]);
        $X = $this->array([-1,1,2,5]);
        $Y = $this->zeros([4],NDArray::int32);
        [$m,$n,$AA,$offsetA,$ldA,$XX,$offsetX,$incX,$right,$YY,$offsetY,$incY] =
            $this->translate_searchsorted($A,$X,false,null,$Y);

        $n = 0;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument n must be greater than 0.');
        $matlib->searchsorted($m,$n,$AA,$offsetA,$ldA,$XX,$offsetX,$incX,$right,$YY,$offsetY,$incY);
    }

    public function testsearchsortedMinusOffsetX()
    {
        $matlib = $this->getMatlib();

        $A = $this->array([0,2,4]);
        $X = $this->array([-1,1,2,5]);
        $Y = $this->zeros([4],NDArray::int32);
        [$m,$n,$AA,$offsetA,$ldA,$XX,$offsetX,$incX,$right,$YY,$offsetY,$incY] =
            $this->translate_searchsorted($A,$X,false,null,$Y);

        $offsetX = -1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument offsetX must be greater than or equals 0.');
        $matlib->searchsorted($m,$n,$AA,$offsetA,$ldA,$XX,$offsetX,$incX,$right,$YY,$offsetY,$incY);
    }

    public function testsearchsortedMinusIncX()
    {
        $matlib = $this->getMatlib();

        $A = $this->array([0,2,4]);
        $X = $this->array([-1,1,2,5]);
        $Y = $this->zeros([4],NDArray::int32);
        [$m,$n,$AA,$offsetA,$ldA,$XX,$offsetX,$incX,$right,$YY,$offsetY,$incY] =
            $this->translate_searchsorted($A,$X,false,null,$Y);

        $incX = 0;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument incX must be greater than 0.');
        $matlib->searchsorted($m,$n,$AA,$offsetA,$ldA,$XX,$offsetX,$incX,$right,$YY,$offsetY,$incY);
    }

    public function testsearchsortedIllegalBufferX()
    {
        $matlib = $this->getMatlib();

        $A = $this->array([0,2,4]);
        $X = $this->array([-1,1,2,5]);
        $Y = $this->zeros([4],NDArray::int32);
        [$m,$n,$AA,$offsetA,$ldA,$XX,$offsetX,$incX,$right,$YY,$offsetY,$incY] =
            $this->translate_searchsorted($A,$X,false,null,$Y);

        $XX = new \stdClass();
        $this->expectException(TypeError::class);
        $this->expectExceptionMessage('must be of type Interop\Polite\Math\Matrix\LinearBuffer');
        $matlib->searchsorted($m,$n,$AA,$offsetA,$ldA,$XX,$offsetX,$incX,$right,$YY,$offsetY,$incY);
    }

    public function testsearchsortedOverflowBufferXwithSize()
    {
        $matlib = $this->getMatlib();

        $A = $this->array([0,2,4]);
        $X = $this->array([-1,1,2,5]);
        $Y = $this->zeros([4],NDArray::int32);
        [$m,$n,$AA,$offsetA,$ldA,$XX,$offsetX,$incX,$right,$YY,$offsetY,$incY] =
            $this->translate_searchsorted($A,$X,false,null,$Y);

        $XX = $this->array([1,2])->buffer();
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Vector specification too large for bufferX');
        $matlib->searchsorted($m,$n,$AA,$offsetA,$ldA,$XX,$offsetX,$incX,$right,$YY,$offsetY,$incY);
    }

    public function testsearchsortedOverflowBufferXwithOffsetX()
    {
        $matlib = $this->getMatlib();

        $A = $this->array([0,2,4]);
        $X = $this->array([-1,1,2,5]);
        $Y = $this->zeros([4],NDArray::int32);
        [$m,$n,$AA,$offsetA,$ldA,$XX,$offsetX,$incX,$right,$YY,$offsetY,$incY] =
            $this->translate_searchsorted($A,$X,false,null,$Y);

        $offsetX = 1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Vector specification too large for bufferX');
        $matlib->searchsorted($m,$n,$AA,$offsetA,$ldA,$XX,$offsetX,$incX,$right,$YY,$offsetY,$incY);
    }

    public function testsearchsortedOverflowBufferXwithIncX()
    {
        $matlib = $this->getMatlib();

        $A = $this->array([0,2,4]);
        $X = $this->array([-1,1,2,5]);
        $Y = $this->zeros([4],NDArray::int32);
        [$m,$n,$AA,$offsetA,$ldA,$XX,$offsetX,$incX,$right,$YY,$offsetY,$incY] =
            $this->translate_searchsorted($A,$X,false,null,$Y);

        $incX = 2;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Vector specification too large for bufferX');
        $matlib->searchsorted($m,$n,$AA,$offsetA,$ldA,$XX,$offsetX,$incX,$right,$YY,$offsetY,$incY);
    }

    public function testsearchsortedMinusOffsetY()
    {
        $matlib = $this->getMatlib();

        $A = $this->array([0,2,4]);
        $X = $this->array([-1,1,2,5]);
        $Y = $this->zeros([4],NDArray::int32);
        [$m,$n,$AA,$offsetA,$ldA,$XX,$offsetX,$incX,$right,$YY,$offsetY,$incY] =
            $this->translate_searchsorted($A,$X,false,null,$Y);

        $offsetY = -1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument offsetY must be greater than or equals 0.');
        $matlib->searchsorted($m,$n,$AA,$offsetA,$ldA,$XX,$offsetX,$incX,$right,$YY,$offsetY,$incY);
    }

    public function testsearchsortedMinusIncY()
    {
        $matlib = $this->getMatlib();

        $A = $this->array([0,2,4]);
        $X = $this->array([-1,1,2,5]);
        $Y = $this->zeros([4],NDArray::int32);
        [$m,$n,$AA,$offsetA,$ldA,$XX,$offsetX,$incX,$right,$YY,$offsetY,$incY] =
            $this->translate_searchsorted($A,$X,false,null,$Y);

        $incY = 0;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument incY must be greater than 0.');
        $matlib->searchsorted($m,$n,$AA,$offsetA,$ldA,$XX,$offsetX,$incX,$right,$YY,$offsetY,$incY);
    }

    public function testsearchsortedIllegalBufferY()
    {
        $matlib = $this->getMatlib();

        $A = $this->array([0,2,4]);
        $X = $this->array([-1,1,2,5]);
        $Y = $this->zeros([4],NDArray::int32);
        [$m,$n,$AA,$offsetA,$ldA,$XX,$offsetX,$incX,$right,$YY,$offsetY,$incY] =
            $this->translate_searchsorted($A,$X,false,null,$Y);

        $YY = new \stdClass();
        $this->expectException(TypeError::class);
        $this->expectExceptionMessage('must be of type Interop\Polite\Math\Matrix\LinearBuffer');
        $matlib->searchsorted($m,$n,$AA,$offsetA,$ldA,$XX,$offsetX,$incX,$right,$YY,$offsetY,$incY);
    }

    public function testsearchsortedOverflowBufferYwithSize()
    {
        $matlib = $this->getMatlib();

        $A = $this->array([0,2,4]);
        $X = $this->array([-1,1,2,5]);
        $Y = $this->zeros([4],NDArray::int32);
        [$m,$n,$AA,$offsetA,$ldA,$XX,$offsetX,$incX,$right,$YY,$offsetY,$incY] =
            $this->translate_searchsorted($A,$X,false,null,$Y);

        $YY = $this->array([1,2])->buffer();
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Vector specification too large for bufferY');
        $matlib->searchsorted($m,$n,$AA,$offsetA,$ldA,$XX,$offsetX,$incX,$right,$YY,$offsetY,$incY);
    }

    public function testsearchsortedOverflowBufferYwithOffsetY()
    {
        $matlib = $this->getMatlib();

        $A = $this->array([0,2,4]);
        $X = $this->array([-1,1,2,5]);
        $Y = $this->zeros([4],NDArray::int32);
        [$m,$n,$AA,$offsetA,$ldA,$XX,$offsetX,$incX,$right,$YY,$offsetY,$incY] =
            $this->translate_searchsorted($A,$X,false,null,$Y);

        $offsetX = 1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Vector specification too large for bufferX');
        $matlib->searchsorted($m,$n,$AA,$offsetA,$ldA,$XX,$offsetX,$incX,$right,$YY,$offsetY,$incY);
    }

    public function testsearchsortedOverflowBufferYwithIncY()
    {
        $matlib = $this->getMatlib();

        $A = $this->array([0,2,4]);
        $X = $this->array([-1,1,2,5]);
        $Y = $this->zeros([4],NDArray::int32);
        [$m,$n,$AA,$offsetA,$ldA,$XX,$offsetX,$incX,$right,$YY,$offsetY,$incY] =
            $this->translate_searchsorted($A,$X,false,null,$Y);

        $incY = 2;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Vector specification too large for bufferY');
        $matlib->searchsorted($m,$n,$AA,$offsetA,$ldA,$XX,$offsetX,$incX,$right,$YY,$offsetY,$incY);
    }

    public function testsearchsortedUnmatchDataType()
    {
        $matlib = $this->getMatlib();

        $A = $this->array([0,2,4],NDArray::float32);
        $X = $this->array([-1,1,2,5],NDArray::float64);
        $Y = $this->zeros([4],NDArray::int32);
        [$m,$n,$AA,$offsetA,$ldA,$XX,$offsetX,$incX,$right,$YY,$offsetY,$incY] =
            $this->translate_searchsorted($A,$X,false,null,$Y);

        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Unmatch data type for A and X');
        $matlib->searchsorted($m,$n,$AA,$offsetA,$ldA,$XX,$offsetX,$incX,$right,$YY,$offsetY,$incY);
    }

    #[DataProvider('providerDtypesFloats')]
    public function testcumsumNormal($params)
    {
        extract($params);
        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3],dtype:$dtype);
        $Y = $this->zeros([3],dtype:$dtype);
        [$n,$XX,$offsetX,$incX,$exclusive,$reverse,$YY,$offsetY,$incY] =
            $this->translate_cumsum($X,false,false,$Y);

        $matlib->cumsum($n,$XX,$offsetX,$incX,$exclusive,$reverse,$YY,$offsetY,$incY);
        $this->assertEquals([1,3,6],$Y->toArray());

        $matlib->cumsum($n,$XX,$offsetX,$incX,true,$reverse,$YY,$offsetY,$incY);
        $this->assertEquals([0,1,3],$Y->toArray());

        $matlib->cumsum($n,$XX,$offsetX,$incX,$exclusive,true,$YY,$offsetY,$incY);
        $this->assertEquals([6,3,1],$Y->toArray());

        $matlib->cumsum($n,$XX,$offsetX,$incX,true,true,$YY,$offsetY,$incY);
        $this->assertEquals([3,1,0],$Y->toArray());

        $X = $this->array([1,NAN,3],dtype:$dtype);
        $Y = $this->zeros([3],dtype:$dtype);
        [$n,$XX,$offsetX,$incX,$exclusive,$reverse,$YY,$offsetY,$incY] =
            $this->translate_cumsum($X,false,false,$Y);

        $matlib->cumsum($n,$XX,$offsetX,$incX,$exclusive,$reverse,$YY,$offsetY,$incY);
        $this->assertEquals(1,$Y[0]);
        $this->assertTrue(is_nan($Y[1]));
        $this->assertTrue(is_nan($Y[2]));

        $matlib->cumsum($n,$XX,$offsetX,$incX,true,$reverse,$YY,$offsetY,$incY);
        $this->assertEquals(0,$Y[0]);
        $this->assertEquals(1,$Y[1]);
        $this->assertTrue(is_nan($Y[2]));

        $matlib->cumsum($n,$XX,$offsetX,$incX,$exclusive,true,$YY,$offsetY,$incY);
        $this->assertTrue(is_nan($Y[0]));
        $this->assertTrue(is_nan($Y[1]));
        $this->assertEquals(1,$Y[2]);
    }

    public function testcumsumMinusN()
    {
        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        $Y = $this->zeros([3],NDArray::float32);
        [$n,$XX,$offsetX,$incX,$exclusive,$reverse,$YY,$offsetY,$incY] =
            $this->translate_cumsum($X,false,false,$Y);

        $n = 0;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument n must be greater than 0.');
        $matlib->cumsum($n,$XX,$offsetX,$incX,$exclusive,$reverse,$YY,$offsetY,$incY);
    }

    public function testcumsumMinusOffsetX()
    {
        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        $Y = $this->zeros([3],NDArray::float32);
        [$n,$XX,$offsetX,$incX,$exclusive,$reverse,$YY,$offsetY,$incY] =
            $this->translate_cumsum($X,false,false,$Y);

        $offsetX = -1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument offsetX must be greater than or equals 0.');
        $matlib->cumsum($n,$XX,$offsetX,$incX,$exclusive,$reverse,$YY,$offsetY,$incY);
    }

    public function testcumsumMinusIncX()
    {
        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        $Y = $this->zeros([3],NDArray::float32);
        [$n,$XX,$offsetX,$incX,$exclusive,$reverse,$YY,$offsetY,$incY] =
            $this->translate_cumsum($X,false,false,$Y);

        $incX = 0;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument incX must be greater than 0.');
        $matlib->cumsum($n,$XX,$offsetX,$incX,$exclusive,$reverse,$YY,$offsetY,$incY);
    }

    public function testcumsumIllegalBufferX()
    {
        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        $Y = $this->zeros([3],NDArray::float32);
        [$n,$XX,$offsetX,$incX,$exclusive,$reverse,$YY,$offsetY,$incY] =
            $this->translate_cumsum($X,false,false,$Y);

        $XX = new \stdClass();
        $this->expectException(TypeError::class);
        $this->expectExceptionMessage('must be of type Interop\Polite\Math\Matrix\LinearBuffer');
        $matlib->cumsum($n,$XX,$offsetX,$incX,$exclusive,$reverse,$YY,$offsetY,$incY);
    }

    public function testcumsumOverflowBufferXwithSize()
    {
        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        $Y = $this->zeros([3],NDArray::float32);
        [$n,$XX,$offsetX,$incX,$exclusive,$reverse,$YY,$offsetY,$incY] =
            $this->translate_cumsum($X,false,false,$Y);

        $XX = $this->array([1,2])->buffer();
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Vector specification too large for bufferX');
        $matlib->cumsum($n,$XX,$offsetX,$incX,$exclusive,$reverse,$YY,$offsetY,$incY);
    }

    public function testcumsumOverflowBufferXwithOffsetX()
    {
        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        $Y = $this->zeros([3],NDArray::float32);
        [$n,$XX,$offsetX,$incX,$exclusive,$reverse,$YY,$offsetY,$incY] =
            $this->translate_cumsum($X,false,false,$Y);

        $offsetX = 1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Vector specification too large for bufferX');
        $matlib->cumsum($n,$XX,$offsetX,$incX,$exclusive,$reverse,$YY,$offsetY,$incY);
    }

    public function testcumsumOverflowBufferXwithIncX()
    {
        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        $Y = $this->zeros([3],NDArray::float32);
        [$n,$XX,$offsetX,$incX,$exclusive,$reverse,$YY,$offsetY,$incY] =
            $this->translate_cumsum($X,false,false,$Y);

        $incX = 2;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Vector specification too large for bufferX');
        $matlib->cumsum($n,$XX,$offsetX,$incX,$exclusive,$reverse,$YY,$offsetY,$incY);
    }

    public function testcumsumMinusOffsetY()
    {
        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        $Y = $this->zeros([3],NDArray::float32);
        [$n,$XX,$offsetX,$incX,$exclusive,$reverse,$YY,$offsetY,$incY] =
            $this->translate_cumsum($X,false,false,$Y);

        $offsetY = -1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument offsetY must be greater than or equals 0.');
        $matlib->cumsum($n,$XX,$offsetX,$incX,$exclusive,$reverse,$YY,$offsetY,$incY);
    }

    public function testcumsumMinusIncY()
    {
        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        $Y = $this->zeros([3],NDArray::float32);
        [$n,$XX,$offsetX,$incX,$exclusive,$reverse,$YY,$offsetY,$incY] =
            $this->translate_cumsum($X,false,false,$Y);

        $incY = 0;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument incY must be greater than 0.');
        $matlib->cumsum($n,$XX,$offsetX,$incX,$exclusive,$reverse,$YY,$offsetY,$incY);
    }

    public function testcumsumIllegalBufferY()
    {
        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        $Y = $this->zeros([3],NDArray::float32);
        [$n,$XX,$offsetX,$incX,$exclusive,$reverse,$YY,$offsetY,$incY] =
            $this->translate_cumsum($X,false,false,$Y);

        $YY = new \stdClass();
        $this->expectException(TypeError::class);
        $this->expectExceptionMessage('must be of type Interop\Polite\Math\Matrix\LinearBuffer');
        $matlib->cumsum($n,$XX,$offsetX,$incX,$exclusive,$reverse,$YY,$offsetY,$incY);
    }

    public function testcumsumOverflowBufferYwithSize()
    {
        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        $Y = $this->zeros([3],NDArray::float32);
        [$n,$XX,$offsetX,$incX,$exclusive,$reverse,$YY,$offsetY,$incY] =
            $this->translate_cumsum($X,false,false,$Y);

        $YY = $this->array([1,2])->buffer();
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Vector specification too large for bufferY');
        $matlib->cumsum($n,$XX,$offsetX,$incX,$exclusive,$reverse,$YY,$offsetY,$incY);
    }

    public function testcumsumOverflowBufferYwithOffsetY()
    {
        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        $Y = $this->zeros([3],NDArray::float32);
        [$n,$XX,$offsetX,$incX,$exclusive,$reverse,$YY,$offsetY,$incY] =
            $this->translate_cumsum($X,false,false,$Y);

        $offsetX = 1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Vector specification too large for bufferX');
        $matlib->cumsum($n,$XX,$offsetX,$incX,$exclusive,$reverse,$YY,$offsetY,$incY);
    }

    public function testcumsumOverflowBufferYwithIncY()
    {
        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        $Y = $this->zeros([3],NDArray::float32);
        [$n,$XX,$offsetX,$incX,$exclusive,$reverse,$YY,$offsetY,$incY] =
            $this->translate_cumsum($X,false,false,$Y);

        $incY = 2;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Vector specification too large for bufferY');
        $matlib->cumsum($n,$XX,$offsetX,$incX,$exclusive,$reverse,$YY,$offsetY,$incY);
    }

    public function testcumsumUnmatchDataType()
    {
        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        $Y = $this->zeros([3],NDArray::float64);
        [$n,$XX,$offsetX,$incX,$exclusive,$reverse,$YY,$offsetY,$incY] =
            $this->translate_cumsum($X,false,false,$Y);

        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Unmatch data type for X and Y');
        $matlib->cumsum($n,$XX,$offsetX,$incX,$exclusive,$reverse,$YY,$offsetY,$incY);
    }

    #[DataProvider('providerDtypesFloats')]
    public function testcumsumbNormal($params)
    {
        extract($params);
        $matlib = $this->getMatlib();

        $X = $this->array([1,2,1,2],dtype:$dtype);
        $X2 = $this->array([1,2,3,4],dtype:$dtype);
        $Y = $this->zeros([4],dtype:$dtype);
        [$m,$n,$k,$XX,$offsetX,$exclusive,$reverse,$YY,$offsetY] =
            $this->translate_cumsumb($X,exclusive:false,reverse:false,outputs:$Y);
        [$m2,$n2,$k2,$XX2,$offsetX2,$exclusive,$reverse,$YY2,$offsetY] =
            $this->translate_cumsumb($X2,exclusive:false,reverse:false,outputs:$Y);

        $matlib->cumsumb($m,$n,$k,$XX,$offsetX,$exclusive,$reverse,$YY,$offsetY);
        $this->assertEquals(
            [1,3,4,6],
            $Y->toArray()
        );

        $matlib->cumsumb($m2,$n2,$k2,$XX2,$offsetX2,$exclusive,$reverse,$YY,$offsetY);
        $this->assertEquals(
            [1,3,6,10],
            $Y->toArray()
        );

        // exclusive=true
        $matlib->cumsumb($m,$n,$k,$XX,$offsetX,true,$reverse,$YY,$offsetY);
        $this->assertEquals(
            [0,1,3,4],
            $Y->toArray()
        );

        $matlib->cumsumb($m2,$n2,$k2,$XX2,$offsetX2,true,$reverse,$YY,$offsetY);
        $this->assertEquals(
            [0, 1, 3, 6],
            $Y->toArray()
        );

        // reverse=true
        $matlib->cumsumb($m,$n,$k,$XX,$offsetX,$exclusive,true,$YY,$offsetY);
        $this->assertEquals(
            [6,5,3,2],
            $Y->toArray()
        );

        $matlib->cumsumb($m2,$n2,$k2,$XX2,$offsetX2,$exclusive,true,$YY,$offsetY);
        $this->assertEquals(
            [10,9,7,4],
            $Y->toArray()
        );

        // exclusive=true & reverse=true
        $matlib->cumsumb($m2,$n2,$k2,$XX2,$offsetX2,true,true,$YY,$offsetY);
        $this->assertEquals(
            [9, 7, 4, 0],
            $Y->toArray()
        );

        // nan data
        $X = $this->array([1,2,NAN,2],dtype:$dtype);
        [$m,$n,$k,$XX,$offsetX,$exclusive,$reverse,$YY,$offsetY] =
            $this->translate_cumsumb($X,exclusive:false,reverse:false,outputs:$Y);
        $matlib->cumsumb($m,$n,$k,$XX,$offsetX,$exclusive,$reverse,$YY,$offsetY);
        $this->assertEquals(1.0,$Y[0]);
        $this->assertEquals(3.0,$Y[1]);
        $this->assertTrue(is_nan($Y[2]));
        $this->assertTrue(is_nan($Y[3]));

        // nan data with reverse
        $X = $this->array([1,2,NAN,2],dtype:$dtype);
        [$m,$n,$k,$XX,$offsetX,$exclusive,$reverse,$YY,$offsetY] =
            $this->translate_cumsumb($X,exclusive:false,reverse:false,outputs:$Y);
        $matlib->cumsumb($m,$n,$k,$XX,$offsetX,$exclusive,true,$YY,$offsetY);
        $this->assertTrue(is_nan($Y[0]));
        $this->assertTrue(is_nan($Y[1]));
        $this->assertTrue(is_nan($Y[2]));
        $this->assertEquals(2.0,$Y[3]);
        //$this->assertEquals(3.0,$Y[2]);
        //$this->assertEquals(1.0,$Y[3]);
    }

    #[DataProvider('providerDtypesFloats')]
    public function testcumsumbWithAxis($params)
    {
        extract($params);
        $matlib = $this->getMatlib();

        $X = $this->array([
            [[ 1, 2, 3],[ 4, 5, 6],[ 7, 8, 9]],
            [[11,12,13],[14,15,16],[17,18,19]],
            [[21,22,23],[24,25,26],[27,28,29]],
        ],dtype:$dtype);
        $Y = $this->zeros([3,3,3],dtype:$dtype);

        [$m,$n,$k,$XX,$offsetX,$exclusive,$reverse,$YY,$offsetY] =
            $this->translate_cumsumb($X,axis:0,exclusive:false,reverse:false,outputs:$Y);
        $matlib->cumsumb($m,$n,$k,$XX,$offsetX,$exclusive,$reverse,$YY,$offsetY);
        $this->assertEquals([
            [[ 1,  2,  3],
             [ 4,  5,  6],
             [ 7,  8,  9]],
    
            [[12, 14, 16],
             [18, 20, 22],
             [24, 26, 28]],
    
            [[33, 36, 39],
             [42, 45, 48],
             [51, 54, 57]]
        ],$Y->toArray());

        [$m,$n,$k,$XX,$offsetX,$exclusive,$reverse,$YY,$offsetY] =
            $this->translate_cumsumb($X,axis:1,exclusive:false,reverse:false,outputs:$Y);
        $matlib->cumsumb($m,$n,$k,$XX,$offsetX,$exclusive,$reverse,$YY,$offsetY);
        $this->assertEquals([
            [[ 1,  2,  3],
             [ 5,  7,  9],
             [12, 15, 18]],
    
            [[11, 12, 13],
             [25, 27, 29],
             [42, 45, 48]],
    
            [[21, 22, 23],
             [45, 47, 49],
             [72, 75, 78]]
        ],$Y->toArray());

        [$m,$n,$k,$XX,$offsetX,$exclusive,$reverse,$YY,$offsetY] =
            $this->translate_cumsumb($X,axis:2,exclusive:false,reverse:false,outputs:$Y);
        $matlib->cumsumb($m,$n,$k,$XX,$offsetX,$exclusive,$reverse,$YY,$offsetY);
        $this->assertEquals([
            [[ 1,  3,  6],
             [ 4,  9, 15],
             [ 7, 15, 24]],
    
            [[11, 23, 36],
             [14, 29, 45],
             [17, 35, 54]],
    
            [[21, 43, 66],
             [24, 49, 75],
             [27, 55, 84]]
        ],$Y->toArray());

        // exclusive
        [$m,$n,$k,$XX,$offsetX,$exclusive,$reverse,$YY,$offsetY] =
            $this->translate_cumsumb($X,axis:1,exclusive:true,reverse:false,outputs:$Y);
        $matlib->cumsumb($m,$n,$k,$XX,$offsetX,$exclusive,$reverse,$YY,$offsetY);
        $this->assertEquals([
            [[ 0,  0,  0],
            [ 1,  2,  3],
            [ 5,  7,  9]],
    
           [[ 0,  0,  0],
            [11, 12, 13],
            [25, 27, 29]],
    
           [[ 0,  0,  0],
            [21, 22, 23],
            [45, 47, 49]]
        ],$Y->toArray());

        // reverse
        [$m,$n,$k,$XX,$offsetX,$exclusive,$reverse,$YY,$offsetY] =
            $this->translate_cumsumb($X,axis:1,exclusive:false,reverse:true,outputs:$Y);
        $matlib->cumsumb($m,$n,$k,$XX,$offsetX,$exclusive,$reverse,$YY,$offsetY);
        $this->assertEquals([
            [[12, 15, 18],
            [11, 13, 15],
            [ 7,  8,  9]],
    
           [[42, 45, 48],
            [31, 33, 35],
            [17, 18, 19]],
    
           [[72, 75, 78],
            [51, 53, 55],
            [27, 28, 29]]
        ],$Y->toArray());

        // exclusive & reverse
        [$m,$n,$k,$XX,$offsetX,$exclusive,$reverse,$YY,$offsetY] =
            $this->translate_cumsumb($X,axis:1,exclusive:true,reverse:true,outputs:$Y);
        $matlib->cumsumb($m,$n,$k,$XX,$offsetX,$exclusive,$reverse,$YY,$offsetY);
        $this->assertEquals([
            [[11, 13, 15],
            [ 7,  8,  9],
            [ 0,  0,  0]],
    
           [[31, 33, 35],
            [17, 18, 19],
            [ 0,  0,  0]],
    
           [[51, 53, 55],
            [27, 28, 29],
            [ 0,  0,  0]]
        ],$Y->toArray());
    }


    #[DataProvider('providerDtypesFloats')]
    public function testzerosNormal($params)
    {
        extract($params);
        $matlib = $this->getMatlib();

        $X = $this->array([1,2,4,9],dtype:$dtype);
        [$N,$XX,$offX,$incX] =
            $this->translate_square($X);

        $matlib->zeros($N,$XX,$offX,$incX);

        $this->assertEquals([0,0,0,0],$X->toArray());
    }

    public function testzerosMinusN()
    {
        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        [$N,$XX,$offX,$incX] =
            $this->translate_square($X);

        $N = 0;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument n must be greater than 0.');
        $matlib->zeros($N,$XX,$offX,$incX);
    }

    public function testzerosMinusOffsetX()
    {
        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        [$N,$XX,$offX,$incX] =
            $this->translate_square($X);

        $offX = -1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument offsetX must be greater than or equals 0.');
        $matlib->zeros($N,$XX,$offX,$incX);
    }

    public function testzerosMinusIncX()
    {
        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        [$N,$XX,$offX,$incX] =
            $this->translate_square($X);

        $incX = 0;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument incX must be greater than 0.');
        $matlib->zeros($N,$XX,$offX,$incX);
    }

    public function testzerosIllegalBufferX()
    {
        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        [$N,$XX,$offX,$incX] =
            $this->translate_square($X);

        $XX = new \stdClass();
        $this->expectException(TypeError::class);
        $this->expectExceptionMessage('must be of type Interop\Polite\Math\Matrix\LinearBuffer');
        $matlib->zeros($N,$XX,$offX,$incX);
    }

    public function testzerosOverflowBufferXwithSize()
    {
        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        [$N,$XX,$offX,$incX] =
            $this->translate_square($X);

        $XX = $this->array([1,2])->buffer();
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Vector specification too large for bufferX');
        $matlib->zeros($N,$XX,$offX,$incX);
    }

    public function testzerosOverflowBufferXwithOffsetX()
    {
        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        [$N,$XX,$offX,$incX] =
            $this->translate_square($X);

        $offX = 1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Vector specification too large for bufferX');
        $matlib->zeros($N,$XX,$offX,$incX);
    }

    public function testzerosOverflowBufferXwithIncX()
    {
        $matlib = $this->getMatlib();

        $X = $this->array([1,2,3]);
        [$N,$XX,$offX,$incX] =
            $this->translate_square($X);

        $incX = 2;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Vector specification too large for bufferX');
        $matlib->zeros($N,$XX,$offX,$incX);
    }

    #[DataProvider('providerDtypesFloatsAndInteger32')]
    public function testTranspose1DNormal($params)
    {
        extract($params);
        $matlib = $this->getMatlib();

        $A = $this->array([1,2,4,9],dtype:$dtype);
        $B = $this->zerosLike($A);
        [
            $sourceShape,
            $permBuf,
            $AA, $offsetA,
            $BB, $offsetB,
        ] = $this->translate_transpose($A,[0],$B);

        $matlib->transpose(
            $sourceShape,
            $permBuf,
            $AA, $offsetA,
            $BB, $offsetB,
        );

        $this->assertEquals([1,2,4,9],$B->toArray());
    }

    #[DataProvider('providerDtypesFloatsAndInteger32')]
    public function testTranspose2DNormal($params)
    {
        extract($params);
        $matlib = $this->getMatlib();

        $A = $this->array([[1,2,3],[4,5,6]],dtype:$dtype);
        $B = $this->zeros([3,2],dtype:$dtype);
        [
            $sourceShape,
            $permBuf,
            $AA, $offsetA,
            $BB, $offsetB,
        ] = $this->translate_transpose($A,[1,0],$B);

        $matlib->transpose(
            $sourceShape,
            $permBuf,
            $AA, $offsetA,
            $BB, $offsetB,
        );

        $this->assertEquals([[1,4],[2,5],[3,6]],$B->toArray());
    }

    #[DataProvider('providerDtypesFloatsAndInteger32')]
    public function testTranspose3DNormal($params)
    {
        extract($params);
        $matlib = $this->getMatlib();

        $A = $this->array([
            [[0,1,2,3],
             [4,5,6,7],
             [8,9,10,11]],
            [[12,13,14,15],
             [16,17,18,19],
             [20,21,22,23]],
        ],dtype:$dtype);
        $B = $this->zeros([4,3,2],dtype:$dtype);
        [
            $sourceShape,
            $permBuf,
            $AA, $offsetA,
            $BB, $offsetB,
        ] = $this->translate_transpose($A,[2,1,0],$B);

        $matlib->transpose(
            $sourceShape,
            $permBuf,
            $AA, $offsetA,
            $BB, $offsetB,
        );

        $this->assertEquals([
           [[ 0., 12.],
            [ 4., 16.],
            [ 8., 20.]],
    
           [[ 1., 13.],
            [ 5., 17.],
            [ 9., 21.]],
    
           [[ 2., 14.],
            [ 6., 18.],
            [10., 22.]],
    
           [[ 3., 15.],
            [ 7., 19.],
            [11., 23.]]            
        ],$B->toArray());
    }

    #[DataProvider('providerDtypesFloatsAndInteger32')]
    public function testTranspose3DWithPerm($params)
    {
        extract($params);
        $matlib = $this->getMatlib();

        $A = $this->array([
            [[0,1,2,3],
             [4,5,6,7],
             [8,9,10,11]],
            [[12,13,14,15],
             [16,17,18,19],
             [20,21,22,23]],
        ],dtype:$dtype);
        $B = $this->zeros([2,4,3],dtype:$dtype);
        [
            $sourceShape,
            $permBuf,
            $AA, $offsetA,
            $BB, $offsetB,
        ] = $this->translate_transpose($A,[0,2,1],$B);

        $matlib->transpose(
            $sourceShape,
            $permBuf,
            $AA, $offsetA,
            $BB, $offsetB,
        );

        $this->assertEquals([
            [[ 0.,  4.,  8.],
             [ 1.,  5.,  9.],
             [ 2.,  6., 10.],
             [ 3.,  7., 11.]],
    
            [[12., 16., 20.],
             [13., 17., 21.],
             [14., 18., 22.],
             [15., 19., 23.]]
        ],$B->toArray());
    }

    public function testTransposeShapeDtypeError()
    {
        $matlib = $this->getMatlib();
    
        $A = $this->array([
            [[0,1,2,3],
             [4,5,6,7],
             [8,9,10,11]],
            [[12,13,14,15],
             [16,17,18,19],
             [20,21,22,23]],
        ],NDArray::int32);
        $B = $this->zeros([2,4,3],NDArray::int32);
        [
            $sourceShape,
            $permBuf,
            $AA, $offsetA,
            $BB, $offsetB,
        ] = $this->translate_transpose($A,[0,2,1],$B);
        $sourceShape = $this->array($A->shape(),NDArray::float32)->buffer();
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('data type of shape buffer must be int32.');
        $matlib->transpose(
            $sourceShape,
            $permBuf,
            $AA, $offsetA,
            $BB, $offsetB,
        );
    }

    public function testTransposeShapeValueError()
    {
        $matlib = $this->getMatlib();
    
        $A = $this->array([
            [[0,1,2,3],
             [4,5,6,7],
             [8,9,10,11]],
            [[12,13,14,15],
             [16,17,18,19],
             [20,21,22,23]],
        ],NDArray::int32);
        $B = $this->zeros([2,4,3],NDArray::int32);
        [
            $sourceShape,
            $permBuf,
            $AA, $offsetA,
            $BB, $offsetB,
        ] = $this->translate_transpose($A,[0,2,1],$B);
        $sourceShape[1] = 0;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('shape values must be greater than 0.');
        $matlib->transpose(
            $sourceShape,
            $permBuf,
            $AA, $offsetA,
            $BB, $offsetB,
        );
    }

    public function testTransposePermSizeError()
    {
        $matlib = $this->getMatlib();
    
        $A = $this->array([
            [[0,1,2,3],
             [4,5,6,7],
             [8,9,10,11]],
            [[12,13,14,15],
             [16,17,18,19],
             [20,21,22,23]],
        ],NDArray::int32);
        $B = $this->zeros([2,4,3],NDArray::int32);
        [
            $sourceShape,
            $permBuf,
            $AA, $offsetA,
            $BB, $offsetB,
        ] = $this->translate_transpose($A,[0,2,1,3],$B);
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('matrix shape and perm must be same size.');
        $matlib->transpose(
            $sourceShape,
            $permBuf,
            $AA, $offsetA,
            $BB, $offsetB,
        );
    }

    public function testTransposePermDtypeError()
    {
        $matlib = $this->getMatlib();
    
        $A = $this->array([
            [[0,1,2,3],
             [4,5,6,7],
             [8,9,10,11]],
            [[12,13,14,15],
             [16,17,18,19],
             [20,21,22,23]],
        ],NDArray::int32);
        $B = $this->zeros([2,4,3],NDArray::int32);
        [
            $sourceShape,
            $permBuf,
            $AA, $offsetA,
            $BB, $offsetB,
        ] = $this->translate_transpose($A,[0,2,1],$B);
        $permBuf = $this->array([0,2,1],NDArray::float32)->buffer();
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('data type of perm buffer must be int32.');
        $matlib->transpose(
            $sourceShape,
            $permBuf,
            $AA, $offsetA,
            $BB, $offsetB,
        );
    }

    public function testTransposeMatrixASizeError()
    {
        $matlib = $this->getMatlib();
    
        $A = $this->array([
            [[0,1,2,3],
             [4,5,6,7],
             [8,9,10,11]],
            [[12,13,14,15],
             [16,17,18,19],
             [20,21,22,23]],
        ],NDArray::int32);
        $B = $this->zeros([2,4,3],NDArray::int32);
        [
            $sourceShape,
            $permBuf,
            $AA, $offsetA,
            $BB, $offsetB,
        ] = $this->translate_transpose($A,[0,2,1],$B);
        $sourceShape[0] = 3;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Vector specification too large for bufferA.');
        $matlib->transpose(
            $sourceShape,
            $permBuf,
            $AA, $offsetA,
            $BB, $offsetB,
        );
    }

    public function testTransposeMatrixBSizeError()
    {
        $matlib = $this->getMatlib();
    
        $A = $this->array([
            [[0,1,2,3],
             [4,5,6,7],
             [8,9,10,11]],
            [[12,13,14,15],
             [16,17,18,19],
             [20,21,22,23]],
        ],NDArray::int32);
        $B = $this->zeros([2,4,3],NDArray::int32);
        [
            $sourceShape,
            $permBuf,
            $AA, $offsetA,
            $BB, $offsetB,
        ] = $this->translate_transpose($A,[0,2,1],$B);
        $BB = $this->zeros([2,4,2],NDArray::int32)->buffer();
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Vector specification too large for bufferB.');
        $matlib->transpose(
            $sourceShape,
            $permBuf,
            $AA, $offsetA,
            $BB, $offsetB,
        );
    }
     
    public function testTransposeMatrixABDtypeError()
    {
        $matlib = $this->getMatlib();
    
        $A = $this->array([
            [[0,1,2,3],
             [4,5,6,7],
             [8,9,10,11]],
            [[12,13,14,15],
             [16,17,18,19],
             [20,21,22,23]],
        ],NDArray::int32);
        $B = $this->zeros([2,4,3],NDArray::float32);
        [
            $sourceShape,
            $permBuf,
            $AA, $offsetA,
            $BB, $offsetB,
        ] = $this->translate_transpose($A,[0,2,1],$B);
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Unmatch data type for A and B.');
        $matlib->transpose(
            $sourceShape,
            $permBuf,
            $AA, $offsetA,
            $BB, $offsetB,
        );
    }

    public function testTransposeDuplicatePerm()
    {
        $matlib = $this->getMatlib();
    
        $A = $this->array([
            [[0,1,2,3],
             [4,5,6,7],
             [8,9,10,11]],
            [[12,13,14,15],
             [16,17,18,19],
             [20,21,22,23]],
        ],NDArray::float32);
        $B = $this->zeros([2,4,3],NDArray::float32);
        [
            $sourceShape,
            $permBuf,
            $AA, $offsetA,
            $BB, $offsetB,
        ] = $this->translate_transpose($A,[0,2,0],$B);
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Perm contained duplicate axis');
        $matlib->transpose(
            $sourceShape,
            $permBuf,
            $AA, $offsetA,
            $BB, $offsetB,
        );
    }

    public function testTransposeOutOfAxisPerm()
    {
        $matlib = $this->getMatlib();
    
        $A = $this->array([
            [[0,1,2,3],
             [4,5,6,7],
             [8,9,10,11]],
            [[12,13,14,15],
             [16,17,18,19],
             [20,21,22,23]],
        ],NDArray::float32);
        $B = $this->zeros([2,4,3],NDArray::float32);
        [
            $sourceShape,
            $permBuf,
            $AA, $offsetA,
            $BB, $offsetB,
        ] = $this->translate_transpose($A,[0,2,3],$B);
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('perm contained an out-of-bounds axis');
        $matlib->transpose(
            $sourceShape,
            $permBuf,
            $AA, $offsetA,
            $BB, $offsetB,
        );
    }

    public function testTransposefloatMatrixAOffset()
    {
        $matlib = $this->getMatlib();
    
        $A = $this->array([
            [[0,0,0,0],
             [0,0,0,0],
             [0,0,0,0]],
            [[0,1,2,3],
             [4,5,6,7],
             [8,9,10,11]],
            [[12,13,14,15],
             [16,17,18,19],
             [20,21,22,23]],
        ],NDArray::float32);
        $A = $A[R(1,3)];
        $B = $this->zeros([4,3,2],NDArray::float32);
        [
            $sourceShape,
            $permBuf,
            $AA, $offsetA,
            $BB, $offsetB,
        ] = $this->translate_transpose($A,[2,1,0],$B);
        $this->assertEquals(3*4,$offsetA);
        $matlib->transpose(
            $sourceShape,
            $permBuf,
            $AA, $offsetA,
            $BB, $offsetB,
        );

        $this->assertEquals([
            [[ 0., 12.],
             [ 4., 16.],
             [ 8., 20.]],
         
            [[ 1., 13.],
             [ 5., 17.],
             [ 9., 21.]],
         
            [[ 2., 14.],
             [ 6., 18.],
             [10., 22.]],
         
            [[ 3., 15.],
             [ 7., 19.],
             [11., 23.]]            
         ],$B->toArray());
    }

    public function testTransposefloatMatrixBOffset()
    {
        $matlib = $this->getMatlib();
    
        $A = $this->array([
            [[0,1,2,3],
             [4,5,6,7],
             [8,9,10,11]],
            [[12,13,14,15],
             [16,17,18,19],
             [20,21,22,23]],
        ],NDArray::float32);
        $origB = $this->zeros([5,3,2],NDArray::float32);
        $B = $origB[R(1,5)];
        [
            $sourceShape,
            $permBuf,
            $AA, $offsetA,
            $BB, $offsetB,
        ] = $this->translate_transpose($A,[2,1,0],$B);
        $matlib->transpose(
            $sourceShape,
            $permBuf,
            $AA, $offsetA,
            $BB, $offsetB,
        );

        $this->assertEquals([
            [[ 0.,  0.],
             [ 0.,  0.],
             [ 0.,  0.]],

            [[ 0., 12.],
             [ 4., 16.],
             [ 8., 20.]],
         
            [[ 1., 13.],
             [ 5., 17.],
             [ 9., 21.]],
         
            [[ 2., 14.],
             [ 6., 18.],
             [10., 22.]],
         
            [[ 3., 15.],
             [ 7., 19.],
             [11., 23.]]            
         ],$origB->toArray());
    }

    public function testTransposeDoubleMatrixAOffset()
    {
        $matlib = $this->getMatlib();
    
        $A = $this->array([
            [[0,0,0,0],
             [0,0,0,0],
             [0,0,0,0]],
            [[0,1,2,3],
             [4,5,6,7],
             [8,9,10,11]],
            [[12,13,14,15],
             [16,17,18,19],
             [20,21,22,23]],
        ],NDArray::float64);
        $A = $A[R(1,3)];
        $B = $this->zeros([4,3,2],NDArray::float64);
        [
            $sourceShape,
            $permBuf,
            $AA, $offsetA,
            $BB, $offsetB,
        ] = $this->translate_transpose($A,[2,1,0],$B);
        $this->assertEquals(3*4,$offsetA);
        $matlib->transpose(
            $sourceShape,
            $permBuf,
            $AA, $offsetA,
            $BB, $offsetB,
        );

        $this->assertEquals([
            [[ 0., 12.],
             [ 4., 16.],
             [ 8., 20.]],
         
            [[ 1., 13.],
             [ 5., 17.],
             [ 9., 21.]],
         
            [[ 2., 14.],
             [ 6., 18.],
             [10., 22.]],
         
            [[ 3., 15.],
             [ 7., 19.],
             [11., 23.]]            
         ],$B->toArray());
    }

    public function testTransposeDoubleMatrixBOffset()
    {
        $matlib = $this->getMatlib();
    
        $A = $this->array([
            [[0,1,2,3],
             [4,5,6,7],
             [8,9,10,11]],
            [[12,13,14,15],
             [16,17,18,19],
             [20,21,22,23]],
        ],NDArray::float64);
        $origB = $this->zeros([5,3,2],NDArray::float64);
        $B = $origB[R(1,5)];
        [
            $sourceShape,
            $permBuf,
            $AA, $offsetA,
            $BB, $offsetB,
        ] = $this->translate_transpose($A,[2,1,0],$B);
        $matlib->transpose(
            $sourceShape,
            $permBuf,
            $AA, $offsetA,
            $BB, $offsetB,
        );

        $this->assertEquals([
            [[ 0.,  0.],
             [ 0.,  0.],
             [ 0.,  0.]],

            [[ 0., 12.],
             [ 4., 16.],
             [ 8., 20.]],
         
            [[ 1., 13.],
             [ 5., 17.],
             [ 9., 21.]],
         
            [[ 2., 14.],
             [ 6., 18.],
             [10., 22.]],
         
            [[ 3., 15.],
             [ 7., 19.],
             [11., 23.]]            
         ],$origB->toArray());
    }

    public function testTransposeintMatrixAOffset()
    {
        $matlib = $this->getMatlib();
    
        $A = $this->array([
            [[0,0,0,0],
             [0,0,0,0],
             [0,0,0,0]],
            [[0,1,2,3],
             [4,5,6,7],
             [8,9,10,11]],
            [[12,13,14,15],
             [16,17,18,19],
             [20,21,22,23]],
        ],NDArray::int32);
        $A = $A[R(1,3)];
        $B = $this->zeros([4,3,2],NDArray::int32);
        [
            $sourceShape,
            $permBuf,
            $AA, $offsetA,
            $BB, $offsetB,
        ] = $this->translate_transpose($A,[2,1,0],$B);
        $this->assertEquals(3*4,$offsetA);
        $matlib->transpose(
            $sourceShape,
            $permBuf,
            $AA, $offsetA,
            $BB, $offsetB,
        );

        $this->assertEquals([
            [[ 0., 12.],
             [ 4., 16.],
             [ 8., 20.]],
         
            [[ 1., 13.],
             [ 5., 17.],
             [ 9., 21.]],
         
            [[ 2., 14.],
             [ 6., 18.],
             [10., 22.]],
         
            [[ 3., 15.],
             [ 7., 19.],
             [11., 23.]]            
         ],$B->toArray());
    }

    public function testTransposeintMatrixBOffset()
    {
        $matlib = $this->getMatlib();
    
        $A = $this->array([
            [[0,1,2,3],
             [4,5,6,7],
             [8,9,10,11]],
            [[12,13,14,15],
             [16,17,18,19],
             [20,21,22,23]],
        ],NDArray::int32);
        $origB = $this->zeros([5,3,2],NDArray::int32);
        $B = $origB[R(1,5)];
        [
            $sourceShape,
            $permBuf,
            $AA, $offsetA,
            $BB, $offsetB,
        ] = $this->translate_transpose($A,[2,1,0],$B);
        $matlib->transpose(
            $sourceShape,
            $permBuf,
            $AA, $offsetA,
            $BB, $offsetB,
        );

        $this->assertEquals([
            [[ 0.,  0.],
             [ 0.,  0.],
             [ 0.,  0.]],

            [[ 0., 12.],
             [ 4., 16.],
             [ 8., 20.]],
         
            [[ 1., 13.],
             [ 5., 17.],
             [ 9., 21.]],
         
            [[ 2., 14.],
             [ 6., 18.],
             [10., 22.]],
         
            [[ 3., 15.],
             [ 7., 19.],
             [11., 23.]]            
         ],$origB->toArray());
    }
    
    #[DataProvider('providerDtypesFloatsAndAllInteger')]
    public function testBandpartNormal($params)
    {
        extract($params);
        $matlib = $this->getMatlib();

        // under
        $A = $this->ones([2,3,3],dtype:$dtype);
        [
            $m,$n,$k,
            $AA, $offsetA,
            $lower,$upper
        ] = $this->translate_bandpart($A,0,-1);
        $matlib->bandpart(
            $m,$n,$k,
            $AA, $offsetA,
            $lower,$upper
        );
        $this->assertEquals([
            [[1,1,1],
             [0,1,1],
             [0,0,1]],
            [[1,1,1],
             [0,1,1],
             [0,0,1]],
        ],$A->toArray());

        $A = $this->ones([2,3,3],dtype:$dtype);
        [
            $m,$n,$k,
            $AA, $offsetA,
            $lower,$upper
        ] = $this->translate_bandpart($A,0,1);
        $matlib->bandpart(
            $m,$n,$k,
            $AA, $offsetA,
            $lower,$upper
        );
        $this->assertEquals([
            [[1,1,0],
             [0,1,1],
             [0,0,1]],
            [[1,1,0],
             [0,1,1],
             [0,0,1]],
        ],$A->toArray());

        // upper
        $A = $this->ones([2,3,3],dtype:$dtype);
        [
            $m,$n,$k,
            $AA, $offsetA,
            $lower,$upper
        ] = $this->translate_bandpart($A,-1,0);
        $matlib->bandpart(
            $m,$n,$k,
            $AA, $offsetA,
            $lower,$upper
        );
        $this->assertEquals([
            [[1,0,0],
             [1,1,0],
             [1,1,1]],
            [[1,0,0],
             [1,1,0],
             [1,1,1]],
        ],$A->toArray());

        $A = $this->ones([2,3,3],dtype:$dtype);
        [
            $m,$n,$k,
            $AA, $offsetA,
            $lower,$upper
        ] = $this->translate_bandpart($A,1,0);
        $matlib->bandpart(
            $m,$n,$k,
            $AA, $offsetA,
            $lower,$upper
        );
        $this->assertEquals([
            [[1,0,0],
             [1,1,0],
             [0,1,1]],
            [[1,0,0],
             [1,1,0],
             [0,1,1]],
        ],$A->toArray());
    }

    public function testBandpartBoolean()
    {
        $dtype = NDArray::bool;
        $matlib = $this->getMatlib();

        // under
        $A = $this->ones([2,3,3],dtype:$dtype);
        [
            $m,$n,$k,
            $AA, $offsetA,
            $lower,$upper
        ] = $this->translate_bandpart($A,0,-1);
        $matlib->bandpart(
            $m,$n,$k,
            $AA, $offsetA,
            $lower,$upper
        );
        $this->assertEquals([
            [[true, true, true],
             [false,true, true],
             [false,false,true]],
            [[true, true, true],
             [false,true, true],
             [false,false,true]],
        ],$A->toArray());

        $A = $this->ones([2,3,3],dtype:$dtype);
        [
            $m,$n,$k,
            $AA, $offsetA,
            $lower,$upper
        ] = $this->translate_bandpart($A,0,1);
        $matlib->bandpart(
            $m,$n,$k,
            $AA, $offsetA,
            $lower,$upper
        );
        $this->assertEquals([
            [[true, true, false],
             [false,true, true],
             [false,false,true]],
            [[true, true, false],
             [false,true, true],
             [false,false,true]],
        ],$A->toArray());

        // upper
        $A = $this->ones([2,3,3],dtype:$dtype);
        [
            $m,$n,$k,
            $AA, $offsetA,
            $lower,$upper
        ] = $this->translate_bandpart($A,-1,0);
        $matlib->bandpart(
            $m,$n,$k,
            $AA, $offsetA,
            $lower,$upper
        );
        $this->assertEquals([
            [[true, false,false],
             [true, true,false],
             [true, true,true ]],
            [[true, false,false],
             [true, true,false],
             [true, true,true ]],
        ],$A->toArray());

        $A = $this->ones([2,3,3],dtype:$dtype);
        [
            $m,$n,$k,
            $AA, $offsetA,
            $lower,$upper
        ] = $this->translate_bandpart($A,1,0);
        $matlib->bandpart(
            $m,$n,$k,
            $AA, $offsetA,
            $lower,$upper
        );
        $this->assertEquals([
            [[true, false,false],
             [true, true, false],
             [false,true, true ]],
            [[true, false,false],
             [true, true, false],
             [false,true, true ]],
        ],$A->toArray());
    }

    public function testBandpartParallel()
    {
        $matlib = $this->getMatlib();

        // m > n
        $A = $this->ones([4,3,3]);
        [
            $m,$n,$k,
            $AA, $offsetA,
            $lower,$upper
        ] = $this->translate_bandpart($A,0,-1);
        $matlib->bandpart(
            $m,$n,$k,
            $AA, $offsetA,
            $lower,$upper
        );
        $this->assertEquals([
            [[1,1,1],
             [0,1,1],
             [0,0,1]],
            [[1,1,1],
             [0,1,1],
             [0,0,1]],
            [[1,1,1],
             [0,1,1],
             [0,0,1]],
            [[1,1,1],
             [0,1,1],
             [0,0,1]],
        ],$A->toArray());

        // m < n
        $A = $this->ones([2,3,3]);
        [
            $m,$n,$k,
            $AA, $offsetA,
            $lower,$upper
        ] = $this->translate_bandpart($A,0,-1);
        $matlib->bandpart(
            $m,$n,$k,
            $AA, $offsetA,
            $lower,$upper
        );
        $this->assertEquals([
            [[1,1,1],
             [0,1,1],
             [0,0,1]],
            [[1,1,1],
             [0,1,1],
             [0,0,1]],
        ],$A->toArray());

    }

    public function testBandpartOffset()
    {
        $matlib = $this->getMatlib();

        $ORGA = $this->ones([2,3,3]);
        $A = $ORGA[R(1,2)];
        [
            $m,$n,$k,
            $AA, $offsetA,
            $lower,$upper
        ] = $this->translate_bandpart($A,0,-1);
        $matlib->bandpart(
            $m,$n,$k,
            $AA, $offsetA,
            $lower,$upper
        );
        $this->assertEquals([
            [[1,1,1],
             [1,1,1],
             [1,1,1]],
            [[1,1,1],
             [0,1,1],
             [0,0,1]],
        ],$ORGA->toArray());
    }

    public function testBandpartOverSize()
    {
        $matlib = $this->getMatlib();

        $A = $this->ones([2,3,3]);
        [
            $m,$n,$k,
            $AA, $offsetA,
            $lower,$upper
        ] = $this->translate_bandpart($A,0,-1);
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Vector specification too large for bufferA');
        $matlib->bandpart(
            $m,$n,$k+1,
            $AA, $offsetA,
            $lower,$upper
        );
    }

    //public function testBandpartUnsupportedDtype()
    //{
    //    $matlib = $this->getMatlib();
//
    //    $A = $this->ones([2,3,3],NDArray::int32);
    //    [
    //        $m,$n,$k,
    //        $AA, $offsetA,
    //        $lower,$upper
    //    ] = $this->translate_bandpart($A,0,-1);
    //    $this->expectException(InvalidArgumentException::class);
    //    $this->expectExceptionMessage('Unsupported data type.');
    //    $matlib->bandpart(
    //        $m,$n,$k,
    //        $AA, $offsetA,
    //        $lower,$upper
    //    );
    //}

    #[DataProvider('providerDtypesFloats')]
    public function testTopkNormal($params)
    {
        extract($params);

        $matlib = $this->getMatlib();

        $m = 2;
        $n = 10;
        $k = 3;
        $sorted = true;

        $input = $this->array([
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,10.0],
            [5.0, 4.0, 3.0, 2.0, 1.0, 5.0, 4.0, 3.0, 2.0, 1.0]
        ],dtype:$dtype);

        $values = $this->zeros([$m,$k],dtype:$dtype);
        $indices = $this->zeros([$m,$k],dtype:NDArray::int32);

        $input_buff     = $input->buffer();
        $input_offset   = $input->offset();
        $values_buff    = $values->buffer();
        $values_offset  = $values->offset();
        $indices_buff   = $indices->buffer();
        $indices_offset = $indices->offset();
        $matlib->topK(
            $m, 
            $n, 
            $input_buff, $input_offset, 
            $k, 
            $sorted, 
            $values_buff, $values_offset, 
            $indices_buff, $indices_offset,
        );

        $this->assertEquals([
            [10.0, 9.0, 8.0],
            [ 5.0, 5.0, 4.0]
        ],$values->toArray());

        $selects = $this->zeros([$m,$k],dtype:$dtype);
        $selects_buff = $selects->buffer();
        $selects_offset   = $selects->offset();
        for($i=0;$i<$m;$i++) {
            $matlib->gather(
                false, // bool $reverse,
                false, // bool $addMode,
                $k,
                1,
                $n,
                $indices_buff, $indices_offset+$i*$k,
                $input_buff, $input_offset+$i*$n,
                $selects_buff, $selects_offset+$i*$k,
            );
        }

        $this->assertEquals([
            [10.0, 9.0, 8.0],
            [ 5.0, 5.0, 4.0]
        ],$selects->toArray());

    }

    #[DataProvider('providerDtypesFloats')]
    public function testTopkLarge($params)
    {
        extract($params);

        $matlib = $this->getMatlib();

        $m = 2;
        $n = 5000;
        $k = 10;
        $sorted = true;

        $input = $this->zeros([$m,$n],dtype:$dtype);
        $matlib->randomUniform(
            $m*$n, $input->buffer(), $input->offset(), 1, 0.0, 1024.0, 0
        );

        $values = $this->zeros([$m,$k],dtype:$dtype);
        $indices = $this->zeros([$m,$k],dtype:NDArray::int32);
        
        $input_buff     = $input->buffer();
        $input_offset   = $input->offset();
        $values_buff    = $values->buffer();
        $values_offset  = $values->offset();
        $indices_buff   = $indices->buffer();
        $indices_offset = $indices->offset();
        $matlib->topK(
            $m, 
            $n, 
            $input_buff, $input_offset, 
            $k, 
            $sorted, 
            $values_buff, $values_offset, 
            $indices_buff, $indices_offset,
        );

        $inputArray = $input->toArray();
        $SortedInput = [];
        foreach($inputArray as $inp) {
            arsort($inp,SORT_NUMERIC);
            $SortedInput[] = $inp;
        }

        $i = 0;
        foreach($SortedInput as $sortedInp) {
            $j = 0;
            foreach ($sortedInp as $topIndex => $topInp) {
                if($topInp!=$values[$i][$j]) {
                    $this->assertEquals($topInp,$values[$i][$j]);
                    break;
                }
                //if($topIndex!=$indices[$i][$j]) {
                //    $this->assertEquals($topIndex,$indices[$i][$j]);
                //    break;
                //}
                if($topInp!=$inputArray[$i][$indices[$i][$j]]) {
                    $this->assertEquals($topIndex,$indices[$i][$j]);
                    break;
                }
                $j++;
                if($j>=$k) {
                    break;
                }
            }
            $i++;
        }
        $this->assertTrue(true);
    }

    #[DataProvider('providerDtypesFloats')]
    public function testTopkWithoutSorted($params)
    {
        extract($params);

        $matlib = $this->getMatlib();

        $m = 2;
        $n = 5000;
        $k = 10;
        $sorted = false;

        $input = $this->zeros([$m,$n],dtype:$dtype);
        $matlib->randomUniform(
            $m*$n, $input->buffer(), $input->offset(), 1, 0.0, 1024.0, 0
        );

        $values = $this->zeros([$m,$k],dtype:$dtype);
        $indices = $this->zeros([$m,$k],dtype:NDArray::int32);
        
        $input_buff     = $input->buffer();
        $input_offset   = $input->offset();
        $values_buff    = $values->buffer();
        $values_offset  = $values->offset();
        $indices_buff   = $indices->buffer();
        $indices_offset = $indices->offset();
        $matlib->topK(
            $m, 
            $n, 
            $input_buff, $input_offset, 
            $k, 
            $sorted, 
            $values_buff, $values_offset, 
            $indices_buff, $indices_offset,
        );

        $inputArray = $input->toArray();
        $SortedInput = [];
        foreach($inputArray as $inp) {
            arsort($inp,SORT_NUMERIC);
            $SortedInput[] = $inp;
        }

        $unmatch = false;
        $i = 0;
        foreach($SortedInput as $sortedInp) {
            $j = 0;
            foreach ($sortedInp as $topIndex => $topInp) {
                if($topInp!=$values[$i][$j]) {
                    $unmatch = true;
                    break;
                }
                //if($topIndex!=$indices[$i][$j]) {
                //    $this->assertEquals($topIndex,$indices[$i][$j]);
                //    break;
                //}
                if($topInp!=$inputArray[$i][$indices[$i][$j]]) {
                    $unmatch = true;
                    break;
                }
                $j++;
                if($j>=$k) {
                    break;
                }
            }
            $i++;
        }
        $this->assertTrue($unmatch);

        $valuesArray = $values->toArray();
        $indicesArray = $indices->toArray();
        $i = 0;
        foreach($SortedInput as $sortedInp) {
            $j = 0;
            foreach ($sortedInp as $topIndex => $topInp) {
                if(!in_array($topInp,$valuesArray[$i])) {
                    $this->assertEquals($topInp,'notfound');
                }
                if(!in_array($topIndex,$indicesArray[$i])) {
                    $this->assertEquals($topIndex,'notfound');
                }
                $j++;
                if($j>=$k) {
                    break;
                }
            }
            $i++;
        }
    }

    #[DataProvider('providerDtypesFloatsAndInteger326w3246indexes')]
    public function testGatherAxisNullNormal($params)
    {
        extract($params);
        $matlib = $this->getMatlib();

        $A = $this->array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]],dtype:$dtype);
        $X = $this->array([0,2],dtype:$indexdtype);
        $B = $this->array([[0,0,0],[0,0,0]],dtype:$dtype);
        [$reduce,$reverse,$addMode,$n,$k,$numClass,$XX,$offX,$AA,$offA,$BB,$offB]
            = $this->translate_gather($scatterAdd=false,$A,$X,$axis=null,$B,$A->dtype());
        $this->assertFalse($reduce);

        $matlib->gather($reverse,$addMode,$n,$k,$numClass,$XX,$offX,$AA,$offA,$BB,$offB);
        $this->assertEquals([[1,2,3],[7,8,9]],$B->toArray());

        $A = $this->array([1,2,3,4],dtype:$dtype);
        $X = $this->array([0,2],dtype:$indexdtype);
        $B = $this->array([0,0],dtype:$dtype);
        [$reduce,$reverse,$addMode,$n,$k,$numClass,$XX,$offX,$AA,$offA,$BB,$offB]
            = $this->translate_gather($scatterAdd=false,$A,$X,$axis=null,$B,$A->dtype());
        $this->assertFalse($reduce);

        $matlib->gather($reverse,$addMode,$n,$k,$numClass,$XX,$offX,$AA,$offA,$BB,$offB);
        $this->assertEquals([1,3],$B->toArray());
    }

    #[DataProvider('providerDtypesFloatsAndInteger326w3246indexes')]
    public function testGatherAxisNullAddMode($params)
    {
        extract($params);
        $matlib = $this->getMatlib();

        $A = $this->array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]],dtype:$dtype);
        $X = $this->array([0,2],dtype:$indexdtype);
        $B = $this->array([[1,1,1],[1,1,1]],dtype:$dtype);
        [$reduce,$reverse,$addMode,$n,$k,$numClass,$XX,$offX,$AA,$offA,$BB,$offB]
            = $this->translate_gather($scatterAdd=false,$A,$X,$axis=null,$B,$A->dtype());
        $this->assertFalse($reduce);
        $addMode = true;

        $matlib->gather($reverse,$addMode,$n,$k,$numClass,$XX,$offX,$AA,$offA,$BB,$offB);
        $this->assertEquals([[2,3,4],[8,9,10]],$B->toArray());

        $A = $this->array([1,2,3,4],dtype:$dtype);
        $X = $this->array([0,2],dtype:$indexdtype);
        $B = $this->array([1,1],dtype:$dtype);
        [$reduce,$reverse,$addMode,$n,$k,$numClass,$XX,$offX,$AA,$offA,$BB,$offB]
            = $this->translate_gather($scatterAdd=false,$A,$X,$axis=null,$B,$A->dtype());
        $this->assertFalse($reduce);
        $addMode = true;

        $matlib->gather($reverse,$addMode,$n,$k,$numClass,$XX,$offX,$AA,$offA,$BB,$offB);
        $this->assertEquals([2,4],$B->toArray());
    }

    #[DataProvider('providerDtypesFloatsAndInteger326w3246indexes')]
    public function testGatherAxisNullReverse($params)
    {
        extract($params);
        $matlib = $this->getMatlib();

        $A = $this->array([[0,0,0],[0,0,0],[0,0,0],[0,0,0]],dtype:$dtype);
        $X = $this->array([0,2],dtype:$indexdtype);
        $B = $this->array([[1,2,3],[7,8,9]],dtype:$dtype);
        [$reduce,$reverse,$addMode,$n,$k,$numClass,$XX,$offX,$AA,$offA,$BB,$offB]
            = $this->translate_gather($scatterAdd=false,$A,$X,$axis=null,$B,$A->dtype());
        $this->assertFalse($reduce);
        $reverse = true;

        $matlib->gather($reverse,$addMode,$n,$k,$numClass,$XX,$offX,$AA,$offA,$BB,$offB);
        $this->assertEquals([[1,2,3],[0,0,0],[7,8,9],[0,0,0]],$A->toArray());

        $A = $this->array([0,0,0,0],dtype:$dtype);
        $X = $this->array([0,2],dtype:$indexdtype);
        $B = $this->array([1,3],dtype:$dtype);
        [$reduce,$reverse,$addMode,$n,$k,$numClass,$XX,$offX,$AA,$offA,$BB,$offB]
            = $this->translate_gather($scatterAdd=false,$A,$X,$axis=null,$B,$A->dtype());
        $this->assertFalse($reduce);
        $reverse = true;

        $matlib->gather($reverse,$addMode,$n,$k,$numClass,$XX,$offX,$AA,$offA,$BB,$offB);
        $this->assertEquals([1,0,3,0],$A->toArray());
    }

    public function testGatherAxisNullLabelNumberOutOfBounds1()
    {
        $matlib = $this->getMatlib();

        $A = $this->array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]],NDArray::float32);
        $X = $this->array([0,4],NDArray::int32);
        $B = $this->array([[0,0,0],[0,0,0]],NDArray::float32);
        [$reduce,$reverse,$addMode,$n,$k,$numClass,$XX,$offX,$AA,$offA,$BB,$offB]
            = $this->translate_gather($scatterAdd=false,$A,$X,$axis=null,$B,$A->dtype());
        $this->assertFalse($reduce);

        $this->expectException(RuntimeException::class);
        $this->expectExceptionMessage('Label number is out of bounds.');
        $matlib->gather($reverse,$addMode,$n,$k,$numClass,$XX,$offX,$AA,$offA,$BB,$offB);
    }

    public function testGatherAxisNullLabelNumberOutOfBounds2()
    {
        $matlib = $this->getMatlib();

        $A = $this->array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]],NDArray::float32);
        $X = $this->array([0,-1],NDArray::int32);
        $B = $this->array([[0,0,0],[0,0,0]],NDArray::float32);
        [$reduce,$reverse,$addMode,$n,$k,$numClass,$XX,$offX,$AA,$offA,$BB,$offB]
            = $this->translate_gather($scatterAdd=false,$A,$X,$axis=null,$B,$A->dtype());
        $this->assertFalse($reduce);

        $this->expectException(RuntimeException::class);
        $this->expectExceptionMessage('Label number is out of bounds.');
        $matlib->gather($reverse,$addMode,$n,$k,$numClass,$XX,$offX,$AA,$offA,$BB,$offB);
    }

    public function testGatherAxisNullMinusN()
    {
        $matlib = $this->getMatlib();

        $A = $this->array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]],NDArray::float32);
        $X = $this->array([0,1],NDArray::int32);
        $B = $this->array([[0,0,0],[0,0,0]],NDArray::float32);
        [$reduce,$reverse,$addMode,$n,$k,$numClass,$XX,$offX,$AA,$offA,$BB,$offB]
            = $this->translate_gather($scatterAdd=false,$A,$X,$axis=null,$B,$A->dtype());
        $this->assertFalse($reduce);

        $n = 0;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument n must be greater than 0.');
        $matlib->gather($reverse,$addMode,$n,$k,$numClass,$XX,$offX,$AA,$offA,$BB,$offB);
    }

    public function testGatherAxisNullMinusK()
    {
        $matlib = $this->getMatlib();

        $A = $this->array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]],NDArray::float32);
        $X = $this->array([0,1],NDArray::int32);
        $B = $this->array([[0,0,0],[0,0,0]],NDArray::float32);
        [$reduce,$reverse,$addMode,$n,$k,$numClass,$XX,$offX,$AA,$offA,$BB,$offB]
            = $this->translate_gather($scatterAdd=false,$A,$X,$axis=null,$B,$A->dtype());
        $this->assertFalse($reduce);

        $k = 0;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument k must be greater than 0.');
        $matlib->gather($reverse,$addMode,$n,$k,$numClass,$XX,$offX,$AA,$offA,$BB,$offB);
    }

    public function testGatherAxisNullMinusNumClass()
    {
        $matlib = $this->getMatlib();

        $A = $this->array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]],NDArray::float32);
        $X = $this->array([0,1],NDArray::int32);
        $B = $this->array([[0,0,0],[0,0,0]],NDArray::float32);
        [$reduce,$reverse,$addMode,$n,$k,$numClass,$XX,$offX,$AA,$offA,$BB,$offB]
            = $this->translate_gather($scatterAdd=false,$A,$X,$axis=null,$B,$A->dtype());
        $this->assertFalse($reduce);

        $numClass = 0;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument numClass must be greater than or equal 0.');
        $matlib->gather($reverse,$addMode,$n,$k,$numClass,$XX,$offX,$AA,$offA,$BB,$offB);
    }

    public function testGatherAxisNullMinusOffsetA()
    {
        $matlib = $this->getMatlib();

        $A = $this->array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]],NDArray::float32);
        $X = $this->array([0,1],NDArray::int32);
        $B = $this->array([[0,0,0],[0,0,0]],NDArray::float32);
        [$reduce,$reverse,$addMode,$n,$k,$numClass,$XX,$offX,$AA,$offA,$BB,$offB]
            = $this->translate_gather($scatterAdd=false,$A,$X,$axis=null,$B,$A->dtype());
        $this->assertFalse($reduce);

        $offA = -1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument offsetA must be greater than or equal 0.');
        $matlib->gather($reverse,$addMode,$n,$k,$numClass,$XX,$offX,$AA,$offA,$BB,$offB);
    }

    public function testGatherAxisNullIllegalBufferA()
    {
        $matlib = $this->getMatlib();

        $A = $this->array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]],NDArray::float32);
        $X = $this->array([0,1],NDArray::int32);
        $B = $this->array([[0,0,0],[0,0,0]],NDArray::float32);
        [$reduce,$reverse,$addMode,$n,$k,$numClass,$XX,$offX,$AA,$offA,$BB,$offB]
            = $this->translate_gather($scatterAdd=false,$A,$X,$axis=null,$B,$A->dtype());
        $this->assertFalse($reduce);

        $AA = new \stdClass();
        $this->expectException(TypeError::class);
        $this->expectExceptionMessage('must be of type Interop\Polite\Math\Matrix\LinearBuffer');
        $matlib->gather($reverse,$addMode,$n,$k,$numClass,$XX,$offX,$AA,$offA,$BB,$offB);
    }

    public function testGatherAxisNullOverflowBufferAwithSize()
    {
        $matlib = $this->getMatlib();

        $A = $this->array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]],NDArray::float32);
        $X = $this->array([0,1],NDArray::int32);
        $B = $this->array([[0,0,0],[0,0,0]],NDArray::float32);
        [$reduce,$reverse,$addMode,$n,$k,$numClass,$XX,$offX,$AA,$offA,$BB,$offB]
            = $this->translate_gather($scatterAdd=false,$A,$X,$axis=null,$B,$A->dtype());
        $this->assertFalse($reduce);

        $AA = $this->array([1,2,3,4,5,6,7,8,9,10,11])->buffer();
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Matrix A specification too large for buffer');
        $matlib->gather($reverse,$addMode,$n,$k,$numClass,$XX,$offX,$AA,$offA,$BB,$offB);
    }

    public function testGatherAxisNullOverflowBufferAwithOffset()
    {
        $matlib = $this->getMatlib();

        $A = $this->array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]],NDArray::float32);
        $X = $this->array([0,1],NDArray::int32);
        $B = $this->array([[0,0,0],[0,0,0]],NDArray::float32);
        [$reduce,$reverse,$addMode,$n,$k,$numClass,$XX,$offX,$AA,$offA,$BB,$offB]
            = $this->translate_gather($scatterAdd=false,$A,$X,$axis=null,$B,$A->dtype());
        $this->assertFalse($reduce);

        $offA = 1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Matrix A specification too large for buffer');
        $matlib->gather($reverse,$addMode,$n,$k,$numClass,$XX,$offX,$AA,$offA,$BB,$offB);
    }

    public function testGatherAxisNullMinusOffsetX()
    {
        $matlib = $this->getMatlib();

        $A = $this->array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]],NDArray::float32);
        $X = $this->array([0,1],NDArray::int32);
        $B = $this->array([[0,0,0],[0,0,0]],NDArray::float32);
        [$reduce,$reverse,$addMode,$n,$k,$numClass,$XX,$offX,$AA,$offA,$BB,$offB]
            = $this->translate_gather($scatterAdd=false,$A,$X,$axis=null,$B,$A->dtype());
        $this->assertFalse($reduce);

        $offX = -1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument offsetX must be greater than or equal 0.');
        $matlib->gather($reverse,$addMode,$n,$k,$numClass,$XX,$offX,$AA,$offA,$BB,$offB);
    }

    public function testGatherAxisNullIllegalBufferX()
    {
        $matlib = $this->getMatlib();

        $A = $this->array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]],NDArray::float32);
        $X = $this->array([0,1],NDArray::int32);
        $B = $this->array([[0,0,0],[0,0,0]],NDArray::float32);
        [$reduce,$reverse,$addMode,$n,$k,$numClass,$XX,$offX,$AA,$offA,$BB,$offB]
            = $this->translate_gather($scatterAdd=false,$A,$X,$axis=null,$B,$A->dtype());
        $this->assertFalse($reduce);

        $XX = new \stdClass();
        $this->expectException(TypeError::class);
        $this->expectExceptionMessage('must be of type Interop\Polite\Math\Matrix\LinearBuffer');
        $matlib->gather($reverse,$addMode,$n,$k,$numClass,$XX,$offX,$AA,$offA,$BB,$offB);
    }

    public function testGatherAxisNullOverflowBufferXwithSize()
    {
        $matlib = $this->getMatlib();

        $A = $this->array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]],NDArray::float32);
        $X = $this->array([0,1],NDArray::int32);
        $B = $this->array([[0,0,0],[0,0,0]],NDArray::float32);
        [$reduce,$reverse,$addMode,$n,$k,$numClass,$XX,$offX,$AA,$offA,$BB,$offB]
            = $this->translate_gather($scatterAdd=false,$A,$X,$axis=null,$B,$A->dtype());
        $this->assertFalse($reduce);

        $XX = $this->array([1])->buffer();
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Matrix X specification too large for buffer.');
        $matlib->gather($reverse,$addMode,$n,$k,$numClass,$XX,$offX,$AA,$offA,$BB,$offB);
    }

    public function testGatherAxisNullOverflowBufferXwithOffsetX()
    {
        $matlib = $this->getMatlib();

        $A = $this->array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]],NDArray::float32);
        $X = $this->array([0,1],NDArray::int32);
        $B = $this->array([[0,0,0],[0,0,0]],NDArray::float32);
        [$reduce,$reverse,$addMode,$n,$k,$numClass,$XX,$offX,$AA,$offA,$BB,$offB]
            = $this->translate_gather($scatterAdd=false,$A,$X,$axis=null,$B,$A->dtype());
        $this->assertFalse($reduce);

        $offX = 1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Matrix X specification too large for buffer.');
        $matlib->gather($reverse,$addMode,$n,$k,$numClass,$XX,$offX,$AA,$offA,$BB,$offB);
    }

    public function testGatherAxisNullMinusOffsetB()
    {
        $matlib = $this->getMatlib();

        $A = $this->array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]],NDArray::float32);
        $X = $this->array([0,1],NDArray::int32);
        $B = $this->array([[0,0,0],[0,0,0]],NDArray::float32);
        [$reduce,$reverse,$addMode,$n,$k,$numClass,$XX,$offX,$AA,$offA,$BB,$offB]
            = $this->translate_gather($scatterAdd=false,$A,$X,$axis=null,$B,$A->dtype());
        $this->assertFalse($reduce);

        $offB = -1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument offsetB must be greater than or equal 0.');
        $matlib->gather($reverse,$addMode,$n,$k,$numClass,$XX,$offX,$AA,$offA,$BB,$offB);
    }

    public function testGatherAxisNullIllegalBufferB()
    {
        $matlib = $this->getMatlib();

        $A = $this->array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]],NDArray::float32);
        $X = $this->array([0,1],NDArray::int32);
        $B = $this->array([[0,0,0],[0,0,0]],NDArray::float32);
        [$reduce,$reverse,$addMode,$n,$k,$numClass,$XX,$offX,$AA,$offA,$BB,$offB]
            = $this->translate_gather($scatterAdd=false,$A,$X,$axis=null,$B,$A->dtype());
        $this->assertFalse($reduce);

        $BB = new \stdClass();
        $this->expectException(TypeError::class);
        $this->expectExceptionMessage('must be of type Interop\Polite\Math\Matrix\LinearBuffer');
        $matlib->gather($reverse,$addMode,$n,$k,$numClass,$XX,$offX,$AA,$offA,$BB,$offB);
    }

    public function testGatherAxisNullOverflowBufferBwithSize()
    {
        $matlib = $this->getMatlib();

        $A = $this->array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]],NDArray::float32);
        $X = $this->array([0,1],NDArray::int32);
        $B = $this->array([[0,0,0],[0,0,0]],NDArray::float32);
        [$reduce,$reverse,$addMode,$n,$k,$numClass,$XX,$offX,$AA,$offA,$BB,$offB]
            = $this->translate_gather($scatterAdd=false,$A,$X,$axis=null,$B,$A->dtype());
        $this->assertFalse($reduce);

        $BB = $this->array([0])->buffer();
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Matrix B specification too large for buffer.');
        $matlib->gather($reverse,$addMode,$n,$k,$numClass,$XX,$offX,$AA,$offA,$BB,$offB);
    }

    public function testGatherAxisNullOverflowBufferBwithOffsetB()
    {
        $matlib = $this->getMatlib();

        $A = $this->array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]],NDArray::float32);
        $X = $this->array([0,1],NDArray::int32);
        $B = $this->array([[0,0,0],[0,0,0]],NDArray::float32);
        [$reduce,$reverse,$addMode,$n,$k,$numClass,$XX,$offX,$AA,$offA,$BB,$offB]
            = $this->translate_gather($scatterAdd=false,$A,$X,$axis=null,$B,$A->dtype());
        $this->assertFalse($reduce);

        $offB = 1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Matrix B specification too large for buffer.');
        $matlib->gather($reverse,$addMode,$n,$k,$numClass,$XX,$offX,$AA,$offA,$BB,$offB);
    }

    #[DataProvider('providerDtypesFloats')]
    public function testReduceGatherAxis1Normal($params)
    {
        extract($params);
        $matlib = $this->getMatlib();

        $A = $this->array([[1,2,3],[4,5,6]],dtype:$dtype);
        $X = $this->array([1,2],dtype:NDArray::int32);
        $B = $this->array([0,0],dtype:$dtype);
        [$reduce,$reverse,$addMode,$m,$n,$numClass,$XX,$offX,$AA,$offA,$BB,$offB]
            = $this->translate_gather($scatterAdd=false,$A,$X,$axis=1,$B,$A->dtype());
        $this->assertTrue($reduce);

        $matlib->reduceGather($reverse,$addMode,$m,$n,$numClass,$XX,$offX,$AA,$offA,$BB,$offB);
        $this->assertEquals([2,6],$B->toArray());
    }

    public function testReduceGatherAxis1LabelNumberOutOfBounds1()
    {
        $matlib = $this->getMatlib();

        $A = $this->array([[1,2,3],[4,5,6]]);
        $X = $this->array([1,3]);
        $B = $this->array([0,0]);
        [$reduce,$reverse,$addMode,$m,$n,$numClass,$XX,$offX,$AA,$offA,$BB,$offB]
            = $this->translate_gather($scatterAdd=false,$A,$X,$axis=1,$B,$A->dtype());
        $this->assertTrue($reduce);

        $this->expectException(RuntimeException::class);
        $this->expectExceptionMessage('Label number is out of bounds.');
        $matlib->reduceGather($reverse,$addMode,$m,$n,$numClass,$XX,$offX,$AA,$offA,$BB,$offB);
    }

    public function testReduceGatherAxis1LabelNumberOutOfBounds2()
    {
        $matlib = $this->getMatlib();

        $A = $this->array([[1,2,3],[4,5,6]]);
        $X = $this->array([1,-1]);
        $B = $this->array([0,0]);
        [$reduce,$reverse,$addMode,$m,$n,$numClass,$XX,$offX,$AA,$offA,$BB,$offB]
            = $this->translate_gather($scatterAdd=false,$A,$X,$axis=1,$B,$A->dtype());
        $this->assertTrue($reduce);

        $this->expectException(RuntimeException::class);
        $this->expectExceptionMessage('Label number is out of bounds.');
        $matlib->reduceGather($reverse,$addMode,$m,$n,$numClass,$XX,$offX,$AA,$offA,$BB,$offB);
    }

    public function testReduceGatherAxis1MinusM()
    {
        $matlib = $this->getMatlib();

        $A = $this->array([[1,2,3],[4,5,6]]);
        $X = $this->array([1,-1]);
        $B = $this->array([0,0]);
        [$reduce,$reverse,$addMode,$m,$n,$numClass,$XX,$offX,$AA,$offA,$BB,$offB]
            = $this->translate_gather($scatterAdd=false,$A,$X,$axis=1,$B,$A->dtype());
        $this->assertTrue($reduce);

        $m = 0;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument m must be greater than 0.');
        $matlib->reduceGather($reverse,$addMode,$m,$n,$numClass,$XX,$offX,$AA,$offA,$BB,$offB);
    }

    public function testReduceGatherAxis1MinusN()
    {
        $matlib = $this->getMatlib();

        $A = $this->array([[1,2,3],[4,5,6]]);
        $X = $this->array([1,-1]);
        $B = $this->array([0,0]);
        [$reduce,$reverse,$addMode,$m,$n,$numClass,$XX,$offX,$AA,$offA,$BB,$offB]
            = $this->translate_gather($scatterAdd=false,$A,$X,$axis=1,$B,$A->dtype());
        $this->assertTrue($reduce);

        $n = 0;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument n must be greater than 0.');
        $matlib->reduceGather($reverse,$addMode,$m,$n,$numClass,$XX,$offX,$AA,$offA,$BB,$offB);
    }

    public function testReduceGatherAxis1MinusOffsetA()
    {
        $matlib = $this->getMatlib();

        $A = $this->array([[1,2,3],[4,5,6]]);
        $X = $this->array([1,-1]);
        $B = $this->array([0,0]);
        [$reduce,$reverse,$addMode,$m,$n,$numClass,$XX,$offX,$AA,$offA,$BB,$offB]
            = $this->translate_gather($scatterAdd=false,$A,$X,$axis=1,$B,$A->dtype());
        $this->assertTrue($reduce);

        $offA = -1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument offsetA must be greater than or equal 0.');
        $matlib->reduceGather($reverse,$addMode,$m,$n,$numClass,$XX,$offX,$AA,$offA,$BB,$offB);
    }

    public function testReduceGatherAxis1IllegalBufferA()
    {
        $matlib = $this->getMatlib();

        $A = $this->array([[1,2,3],[4,5,6]]);
        $X = $this->array([1,-1]);
        $B = $this->array([0,0]);
        [$reduce,$reverse,$addMode,$m,$n,$numClass,$XX,$offX,$AA,$offA,$BB,$offB]
            = $this->translate_gather($scatterAdd=false,$A,$X,$axis=1,$B,$A->dtype());
        $this->assertTrue($reduce);

        $AA = new \stdClass();
        $this->expectException(TypeError::class);
        $this->expectExceptionMessage('must be of type Interop\Polite\Math\Matrix\LinearBuffer');
        $matlib->reduceGather($reverse,$addMode,$m,$n,$numClass,$XX,$offX,$AA,$offA,$BB,$offB);
    }

    public function testReduceGatherAxis1OverflowBufferAwithSize()
    {
        $matlib = $this->getMatlib();

        $A = $this->array([[1,2,3],[4,5,6]]);
        $X = $this->array([1,-1]);
        $B = $this->array([0,0]);
        [$reduce,$reverse,$addMode,$m,$n,$numClass,$XX,$offX,$AA,$offA,$BB,$offB]
            = $this->translate_gather($scatterAdd=false,$A,$X,$axis=1,$B,$A->dtype());
        $this->assertTrue($reduce);

        $AA = $this->array([1,2,3,4,5])->buffer();
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Matrix A specification too large for buffer.');
        $matlib->reduceGather($reverse,$addMode,$m,$n,$numClass,$XX,$offX,$AA,$offA,$BB,$offB);
    }

    public function testReduceGatherAxis1OverflowBufferAwithOffset()
    {
        $matlib = $this->getMatlib();

        $A = $this->array([[1,2,3],[4,5,6]]);
        $X = $this->array([1,-1]);
        $B = $this->array([0,0]);
        [$reduce,$reverse,$addMode,$m,$n,$numClass,$XX,$offX,$AA,$offA,$BB,$offB]
            = $this->translate_gather($scatterAdd=false,$A,$X,$axis=1,$B,$A->dtype());
        $this->assertTrue($reduce);

        $offA = 1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Matrix A specification too large for buffer.');
        $matlib->reduceGather($reverse,$addMode,$m,$n,$numClass,$XX,$offX,$AA,$offA,$BB,$offB);
    }

    public function testReduceGatherAxis1MinusOffsetX()
    {
        $matlib = $this->getMatlib();

        $A = $this->array([[1,2,3],[4,5,6]]);
        $X = $this->array([1,-1]);
        $B = $this->array([0,0]);
        [$reduce,$reverse,$addMode,$m,$n,$numClass,$XX,$offX,$AA,$offA,$BB,$offB]
            = $this->translate_gather($scatterAdd=false,$A,$X,$axis=1,$B,$A->dtype());
        $this->assertTrue($reduce);

        $offX = -1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument offsetX must be greater than or equal 0.');
        $matlib->reduceGather($reverse,$addMode,$m,$n,$numClass,$XX,$offX,$AA,$offA,$BB,$offB);
    }

    public function testReduceGatherAxis1IllegalBufferX()
    {
        $matlib = $this->getMatlib();

        $A = $this->array([[1,2,3],[4,5,6]]);
        $X = $this->array([1,-1]);
        $B = $this->array([0,0]);
        [$reduce,$reverse,$addMode,$m,$n,$numClass,$XX,$offX,$AA,$offA,$BB,$offB]
            = $this->translate_gather($scatterAdd=false,$A,$X,$axis=1,$B,$A->dtype());
        $this->assertTrue($reduce);

        $XX = new \stdClass();
        $this->expectException(TypeError::class);
        $this->expectExceptionMessage('must be of type Interop\Polite\Math\Matrix\LinearBuffer');
        $matlib->reduceGather($reverse,$addMode,$m,$n,$numClass,$XX,$offX,$AA,$offA,$BB,$offB);
    }

    public function testReduceGatherAxis1OverflowBufferXwithSize()
    {
        $matlib = $this->getMatlib();

        $A = $this->array([[1,2,3],[4,5,6]]);
        $X = $this->array([1,-1]);
        $B = $this->array([0,0]);
        [$reduce,$reverse,$addMode,$m,$n,$numClass,$XX,$offX,$AA,$offA,$BB,$offB]
            = $this->translate_gather($scatterAdd=false,$A,$X,$axis=1,$B,$A->dtype());
        $this->assertTrue($reduce);
        $XX = $this->array([1])->buffer();
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Matrix X specification too large for buffer.');
        $matlib->reduceGather($reverse,$addMode,$m,$n,$numClass,$XX,$offX,$AA,$offA,$BB,$offB);
    }

    public function testReduceGatherAxis1OverflowBufferXwithOffsetX()
    {
        $matlib = $this->getMatlib();

        $A = $this->array([[1,2,3],[4,5,6]]);
        $X = $this->array([1,-1]);
        $B = $this->array([0,0]);
        [$reduce,$reverse,$addMode,$m,$n,$numClass,$XX,$offX,$AA,$offA,$BB,$offB]
            = $this->translate_gather($scatterAdd=false,$A,$X,$axis=1,$B,$A->dtype());
        $this->assertTrue($reduce);

        $offX = 1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Matrix X specification too large for buffer.');
        $matlib->reduceGather($reverse,$addMode,$m,$n,$numClass,$XX,$offX,$AA,$offA,$BB,$offB);
    }

    public function testReduceGatherAxis1MinusOffsetB()
    {
        $matlib = $this->getMatlib();

        $A = $this->array([[1,2,3],[4,5,6]]);
        $X = $this->array([1,-1]);
        $B = $this->array([0,0]);
        [$reduce,$reverse,$addMode,$m,$n,$numClass,$XX,$offX,$AA,$offA,$BB,$offB]
            = $this->translate_gather($scatterAdd=false,$A,$X,$axis=1,$B,$A->dtype());
        $this->assertTrue($reduce);

        $offB = -1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument offsetB must be greater than or equal 0.');
        $matlib->reduceGather($reverse,$addMode,$m,$n,$numClass,$XX,$offX,$AA,$offA,$BB,$offB);
    }

    public function testReduceGatherAxis1IllegalBufferB()
    {
        $matlib = $this->getMatlib();

        $A = $this->array([[1,2,3],[4,5,6]]);
        $X = $this->array([1,-1]);
        $B = $this->array([0,0]);
        [$reduce,$reverse,$addMode,$m,$n,$numClass,$XX,$offX,$AA,$offA,$BB,$offB]
            = $this->translate_gather($scatterAdd=false,$A,$X,$axis=1,$B,$A->dtype());
        $this->assertTrue($reduce);

        $BB = new \stdClass();
        $this->expectException(TypeError::class);
        $this->expectExceptionMessage('must be of type Interop\Polite\Math\Matrix\LinearBuffer');
        $matlib->reduceGather($reverse,$addMode,$m,$n,$numClass,$XX,$offX,$AA,$offA,$BB,$offB);
    }

    public function testReduceGatherAxis1OverflowBufferBwithSize()
    {
        $matlib = $this->getMatlib();

        $A = $this->array([[1,2,3],[4,5,6]]);
        $X = $this->array([1,-1]);
        $B = $this->array([0,0]);
        [$reduce,$reverse,$addMode,$m,$n,$numClass,$XX,$offX,$AA,$offA,$BB,$offB]
            = $this->translate_gather($scatterAdd=false,$A,$X,$axis=1,$B,$A->dtype());
        $this->assertTrue($reduce);

        $BB = $this->array([0])->buffer();
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Matrix B specification too large for buffer.');
        $matlib->reduceGather($reverse,$addMode,$m,$n,$numClass,$XX,$offX,$AA,$offA,$BB,$offB);
    }

    public function testReduceGatherAxis1OverflowBufferXwithOffsetB()
    {
        $matlib = $this->getMatlib();

        $A = $this->array([[1,2,3],[4,5,6]]);
        $X = $this->array([1,-1]);
        $B = $this->array([0,0]);
        [$reduce,$reverse,$addMode,$m,$n,$numClass,$XX,$offX,$AA,$offA,$BB,$offB]
            = $this->translate_gather($scatterAdd=false,$A,$X,$axis=1,$B,$A->dtype());
        $this->assertTrue($reduce);

        $offB = 1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Matrix B specification too large for buffer.');
        $matlib->reduceGather($reverse,$addMode,$m,$n,$numClass,$XX,$offX,$AA,$offA,$BB,$offB);
    }
/*
    public function testGatherbNormal()
    {
        $matlib = $this->getMatlib();

        // axis = 0
        // 1D indices
        // A:   2D [m,numClass]
        // X:   1D [m]
        // res: 1D [m]
        $a = $this->array([
            [ 0, 0, 3],
            [ 0, 0, 4],
            [ 0, 2, 0],
            [ 1, 0, 0]]);
        $b = $this->zeros([4]);
        $x = $this->array([2,2,1,0],dtype:NDArray::int32);
        $matlib->gatherb(...$this->translate_gatherb($a,$x,axis:0,output:$b));
        $this->assertEquals([4,3],$a->shape());
        $this->assertEquals([4],$x->shape());
        $this->assertEquals([4],$b->shape()); // reduction axis1
        $trues = $this->array([3,4,2,1]);
        $this->assertEquals($trues->toArray(),$b->toArray());
        
        // axis = 0
        // 1D indices
        // A:   3D [m,numClass,k]
        // X:   1D [m]
        // res: 2D [m,k]
        $a = $this->array([
            [ [0,0], [0,0], [3,3]],
            [ [0,0], [0,0], [4,4]],
            [ [0,0], [2,2], [0,0]],
            [ [1,1], [0,0], [0,0]]]);
        $x = $this->array([2,2,1,0],dtype:NDArray::int32);
        $b = $this->zeros([4,2]);
        $matlib->gatherb(...$this->translate_gatherb($a,$x,axis:0,output:$b));
        $this->assertEquals([4,3,2],$a->shape());
        $this->assertEquals([4],$x->shape());
        $this->assertEquals([4,2],$b->shape()); // reduction axis1
        $trues = $this->array([[3,3],[4,4],[2,2],[1,1]]);
        $this->assertEquals($trues->toArray(),$b->toArray());

        // axis = 1
        // 1D indices
        // A:   3D [m,n,numClass]
        // X:   1D [m]
        // res: 2D [m,n]
        $a = $this->array([
            [[0, 0, 3],
             [0, 0,13]],
            [[0, 0, 4],
             [ 0, 0,14]],
            [[ 0, 2, 0],
             [ 0,12, 0]],
            [[ 1, 0, 0],
             [11, 0, 0]],
        ]);
        $x = $this->array([2,2,1,0],dtype:NDArray::int32);
        $b = $this->zeros([4,2]);
        $matlib->gatherb(...$this->translate_gatherb($a,$x,axis:1,output:$b));
        $this->assertEquals([4,2,3],$a->shape());
        $this->assertEquals([4],$x->shape());
        $this->assertEquals([4,2],$b->shape()); // reduction axis1
        $trues = $this->array([[3,13],[4,14],[2,12],[1,11]]);
        $this->assertEquals($trues->toArray(),$b->toArray());

        // axis = 1
        // 1D indices
        // A:   4D [m,n,numClass,k]
        // X:   1D [m]
        // res: 3D [m,n,k]
        $a = $this->array([
            [[[ 0, 0],[ 0, 0],[ 3, 3]],
             [[ 0, 0],[ 0, 0],[13,13]]],
            [[[ 0, 0],[ 0, 0],[ 4, 4]],
             [[ 0, 0],[ 0, 0],[14,14]]],
            [[[ 0, 0],[ 2, 2],[ 0, 0]],
             [[ 0, 0],[12,12],[ 0, 0]]],
            [[[ 1, 1],[ 0, 0],[ 0, 0]],
             [[11,11],[ 0, 0],[ 0, 0]]],
        ]);
        $x = $this->array([2,2,1,0],dtype:NDArray::int32);
        $b = $this->zeros([4,2,2]);
        $matlib->gatherb(...$this->translate_gatherb($a,$x,axis:1,output:$b));
        $this->assertEquals([4,2,3,2],$a->shape());
        $this->assertEquals([4],$x->shape());
        $this->assertEquals([4,2,2],$b->shape()); // reduction axis1
        $trues = $this->array([[[3,3],[13,13]],[[4,4],[14,14]],[[2,2],[12,12]],[[1,1],[11,11]]]);
        $this->assertEquals($trues->toArray(),$b->toArray());
    }
*/
    public function testGatherbNormal()
    {
        $matlib = $this->getMatlib();

        // axis = 0
        // A:   1D [numClass]
        // X:   1D [n]
        // res: 1D [n]
        $shapeA = [4];
        $a = $this->array(range(0,array_product($shapeA)-1))->reshape($shapeA);
        $x = $this->array([2,2,1],dtype:NDArray::int32);
        $b = $this->zeros([3]);
        $matlib->gatherb(...$this->translate_gatherb($a,$x,outputs:$b));
        $this->assertEquals([4],$a->shape());
        $this->assertEquals([3],$x->shape());
        $this->assertEquals([3],$b->shape()); // reduction axis1
        $trues = $this->array([2,2,1]);
        $this->assertEquals($trues->toArray(),$b->toArray());
        
        // batchDims = 1
        // axis = 1
        // A:   2D [batches,numClass]
        // X:   1D [batches]
        // res: 2D [batches]
        $shapeA = [4,3];
        $a = $this->array(range(0,array_product($shapeA)-1))->reshape($shapeA);
        $x = $this->array([2,2,1,0],dtype:NDArray::int32);
        $b = $this->zeros([4]);
        $matlib->gatherb(...$this->translate_gatherb($a,$x,axis:1,batchDims:1,outputs:$b));
        $this->assertEquals([4,3],$a->shape());
        $this->assertEquals([4],$x->shape());
        $this->assertEquals([4],$b->shape()); // reduction axis1
        $trues = $this->array([2,5,7,9]);
        $this->assertEquals($trues->toArray(),$b->toArray());

        // batchDims = 1
        // axis = 1
        // A:   3D [bathces,numClass,len]
        // X:   1D [bathces]
        // res: 2D [bathces,len]
        $shapeA = [4,3,2];
        $a = $this->array(range(0,array_product($shapeA)-1))->reshape($shapeA);
        $x = $this->array([2,2,1,0],dtype:NDArray::int32);
        $b = $this->zeros([4,2]);
        $matlib->gatherb(...$this->translate_gatherb($a,$x,axis:1,batchDims:1,outputs:$b));
        $this->assertEquals([4,3,2],$a->shape());
        $this->assertEquals([4],$x->shape());
        $this->assertEquals([4,2],$b->shape()); // reduction axis1
        $trues = $this->array([
            [4,5],
            [10,11],
            [14,15],
            [18,19],
        ]);
        $this->assertEquals($trues->toArray(),$b->toArray());

        // for argmax
        // batchDims = 1    // batchDims=axis
        // axis = 1         // default: axis=batchDims
        // detailDepth = 3  // detailDepth=params->ndim()
        // indexDepth = 1   // indexDepth=batchDims
        // A:   2D [batches,numClass,k]
        // X:   2D [batches,k]
        // res: 2D [batches,k]
        $shapeA = [3,2,4];
        $a = $this->array(range(0,array_product($shapeA)-1))->reshape($shapeA);
        $x = $this->array([
            [1,1,1,1],
            [1,1,1,1],
            [1,1,1,1],
        ],dtype:NDArray::int32);
        $b = $this->zeros([3,4]);
        $matlib->gatherb(...$this->translate_gatherb($a,$x,batchDims:1,detailDepth:$a->ndim(),indexDepth:1,outputs:$b));
        $this->assertEquals([3,2,4],$a->shape());
        $this->assertEquals([3,4],$x->shape());
        $this->assertEquals([3,4],$b->shape()); // reduction axis1
        $trues = $this->array([
            [ 4, 5, 6, 7],
            [12,13,14,15],
            [20,21,22,23],
        ]);
        //echo $mo->toString($b,null,true)."\n";
        //$max = $la->reduceMax($a,axis:1);
        //echo $mo->toString($max,null,true)."\n";
        //$max = $la->reduceArgMax($a,axis:1);
        //echo $mo->toString($max,null,true)."\n";
        //$this->assertEquals($max->toArray(),$x->toArray());
        //$max = $la->reduceMax($a,axis:1);
        //echo $mo->toString($max,null,true)."\n";
        //$this->assertEquals($max->toArray(),$trues->toArray());
        $this->assertEquals($trues->toArray(),$b->toArray());

        // batchDims = 1
        // axis = 2
        // A:   3D [batches,m,numClass]
        // X:   1D [batches]
        // res: 2D [batches,m]
        $shapeA = [4,2,3];
        $a = $this->array(range(0,array_product($shapeA)-1))->reshape($shapeA);
        $x = $this->array([2,2,1,0],dtype:NDArray::int32);
        $b = $this->zeros([4,2]);
        $matlib->gatherb(...$this->translate_gatherb($a,$x,axis:2,batchDims:1,outputs:$b));
        $this->assertEquals([4,2,3],$a->shape());
        $this->assertEquals([4],$x->shape());
        $this->assertEquals([4,2],$b->shape()); // reduction axis1
        $trues = $this->array([
            [ 2, 5],
            [ 8,11],
            [13,16],
            [18,21],
        ]);
        $this->assertEquals($trues->toArray(),$b->toArray());

        // batchDims = 1
        // axis = 2
        // A:   4D [batches,m,numClass,len]
        // X:   1D [batches]
        // res: 3D [batches,m,len]
        $shapeA = [4,2,3,2];
        $a = $this->array(range(0,array_product($shapeA)-1))->reshape($shapeA);
        $x = $this->array([2,2,1,0],dtype:NDArray::int32);
        $b = $this->zeros([4,2,2]);
        $matlib->gatherb(...$this->translate_gatherb($a,$x,axis:2,batchDims:1,outputs:$b));
        $this->assertEquals([4,2,3,2],$a->shape());
        $this->assertEquals([4],$x->shape());
        $this->assertEquals([4,2,2],$b->shape()); // reduction axis1
        $trues = $this->array([
            [[ 4, 5],[10,11]],
            [[16,17],[22,23]],
            [[26,27],[32,33]],
            [[36,37],[42,43]],
        ]);
        $this->assertEquals($trues->toArray(),$b->toArray());

        // batchDims = 1
        // axis = 1
        // A:   2D [batches,numClass]
        // X:   2D [batches,n]
        // res: 2D [batches,n]
        $shapeA = [4,3];
        $a = $this->array(range(0,array_product($shapeA)-1))->reshape($shapeA);
        $x = $this->array([
            [0,1],
            [1,0],
            [2,2],
            [1,0]
        ],dtype:NDArray::int32);
        $b = $this->zeros([4,2]);
        $matlib->gatherb(...$this->translate_gatherb($a,$x,axis:1,batchDims:1,outputs:$b));
        $this->assertEquals([4,3],$a->shape());
        $this->assertEquals([4,2],$x->shape());
        $this->assertEquals([4,2],$b->shape()); // reduction axis1
        $trues = $this->array([
            [0,1],
            [4,3],
            [8,8],
            [10,9]
        ]);
        $this->assertEquals($trues->toArray(),$b->toArray());

        // batchDims = 1
        // axis = 1
        // detailDepth = 3
        // indexDepth = 2
        // A:   3D [batches,numClass,k]
        // X:   3D [batches,n,k]
        // res: 3D [batches,n,k]
        $shapeA = [4,5,2];
        $a = $this->array(range(0,array_product($shapeA)-1))->reshape($shapeA);
        $x = $this->array([
            [[0,1],[0,1],[0,1]],
            [[1,0],[1,0],[1,0]],
            [[2,2],[2,2],[2,2]],
            [[1,0],[1,0],[1,0]]
        ],dtype:NDArray::int32);
        $b = $this->zeros([4,3,2]);
        $matlib->gatherb(...$this->translate_gatherb($a,$x,axis:1,batchDims:1,detailDepth:3,indexDepth:2,outputs:$b));
        $this->assertEquals([4,5,2],$a->shape());
        $this->assertEquals([4,3,2],$x->shape());
        $this->assertEquals([4,3,2],$b->shape()); // reduction axis1
        $trues = $this->array([
            [[ 0, 3],[ 0, 3],[ 0, 3]],
            [[12,11],[12,11],[12,11]],
            [[24,25],[24,25],[24,25]],
            [[32,31],[32,31],[32,31]]
        ]);
        $this->assertEquals($trues->toArray(),$b->toArray());

        // batchDims = 1
        // axis = 1
        // detailDepth = 3
        // indexDepth = 2
        // A:   4D [batches,numClass,k,len]
        // X:   3D [batches,n,k]
        // res: 4D [batches,n,k,len]
        $shapeA = [4,5,2,3];
        $a = $this->array(range(0,array_product($shapeA)-1))->reshape($shapeA);
        $x = $this->array([
            [[0,1],[0,1],[0,1]],
            [[1,0],[1,0],[1,0]],
            [[2,2],[2,2],[2,2]],
            [[1,0],[1,0],[1,0]],
        ],dtype:NDArray::int32);
        $b = $this->zeros([4,3,2,3]);
        $matlib->gatherb(...$this->translate_gatherb($a,$x,axis:1,batchDims:1,detailDepth:3,indexDepth:2,outputs:$b));
        $this->assertEquals([4,5,2,3],$a->shape());
        $this->assertEquals([4,3,2],$x->shape());
        $this->assertEquals([4,3,2,3],$b->shape()); // reduction axis1
        $trues = $this->array([
            [[[ 0, 1, 2],[ 9,10,11]],
             [[ 0, 1, 2],[ 9,10,11]],
             [[ 0, 1, 2],[ 9,10,11]],],
            [[[36,37,38],[33,34,35]],
             [[36,37,38],[33,34,35]],
             [[36,37,38],[33,34,35]],],
            [[[72,73,74],[75,76,77]],
             [[72,73,74],[75,76,77]],
             [[72,73,74],[75,76,77]],],
            [[[96,97,98],[93,94,95]],
             [[96,97,98],[93,94,95]],
             [[96,97,98],[93,94,95]],]
        ]);
        $this->assertEquals($trues->toArray(),$b->toArray());

        // batchDims = 1
        // axis = 2
        // detailDepth = 4
        // indexDepth = 2
        // A:   5D [batches,m,numClass,k,len]
        // X:   3D [batches,n,k]
        // res: 5D [batches,m,n,k,len]
        $shapeA = [4,2,5,2,3];
        $a = $this->array(range(0,array_product($shapeA)-1))->reshape($shapeA);
        $x = $this->array([
            [[0,1],[0,1],[0,1]],
            [[1,0],[1,0],[1,0]],
            [[2,2],[2,2],[2,2]],
            [[1,0],[1,0],[1,0]],
        ],dtype:NDArray::int32);
        $b = $this->zeros([4,2,3,2,3]);
        $matlib->gatherb(...$this->translate_gatherb($a,$x,axis:2,batchDims:1,detailDepth:4,indexDepth:2,outputs:$b));
        $this->assertEquals([4,2,5,2,3],$a->shape());
        $this->assertEquals([4,3,2],$x->shape());
        $this->assertEquals([4,2,3,2,3],$b->shape()); // reduction axis1
        $trues = $this->array([
            [[[[ 0, 1, 2],[ 9,10,11]],
              [[ 0, 1, 2],[ 9,10,11]],
              [[ 0, 1, 2],[ 9,10,11]]],
             [[[30,31,32],[39,40,41]],
              [[30,31,32],[39,40,41]],
              [[30,31,32],[39,40,41]]]],
            [[[[66,67,68],[63,64,65]],
              [[66,67,68],[63,64,65]],
              [[66,67,68],[63,64,65]]],
             [[[96,97,98],[93,94,95]],
              [[96,97,98],[93,94,95]],
              [[96,97,98],[93,94,95]]]],
            [[[[132,133,134],[135,136,137]],
              [[132,133,134],[135,136,137]],
              [[132,133,134],[135,136,137]]],
             [[[162,163,164],[165,166,167]],
              [[162,163,164],[165,166,167]],
              [[162,163,164],[165,166,167]]]],
            [[[[186,187,188],[183,184,185]],
              [[186,187,188],[183,184,185]],
              [[186,187,188],[183,184,185]]],
             [[[216,217,218],[213,214,215]],
              [[216,217,218],[213,214,215]],
              [[216,217,218],[213,214,215]]]]
        ]);
        $this->assertEquals($trues->toArray(),$b->toArray());
    }

    public function testScatterbNormal()
    {
        $matlib = $this->getMatlib();

        // axis = 0
        // A:   1D [n]
        // X:   1D [n]
        // res: 1D [numClass]
        $shapeA = [3];
        $a = $this->array([2,2,1]);
        $x = $this->array([2,2,1],dtype:NDArray::int32);
        $shapeB = [4];
        $b = $this->zeros($shapeB);
        $matlib->gatherb(...$this->translate_scatterb($x,$a,$shapeB,outputs:$b));
        $this->assertEquals([3],$a->shape());
        $this->assertEquals([3],$x->shape());
        $this->assertEquals([4],$b->shape());
        $trues = $this->array([0,1,2,0]);
        $this->assertEquals($trues->toArray(),$b->toArray());
        
        // batchDims = 1
        // axis = 1
        // A:   2D [batches]
        // X:   1D [batches]
        // res: 2D [batches,numClass]
        $shapeA = [4];
        $a = $this->array([2,5,7,9]);
        $x = $this->array([2,2,1,0],dtype:NDArray::int32);
        $shapeB = [4,3];
        $b = $this->zeros($shapeB);
        $matlib->gatherb(...$this->translate_scatterb($x,$a,$shapeB,axis:1,batchDims:1,outputs:$b));
        $this->assertEquals([4],$a->shape());
        $this->assertEquals([4],$x->shape());
        $this->assertEquals([4,3],$b->shape());
        $trues = $this->array([
            [0,0,2],
            [0,0,5],
            [0,7,0],
            [9,0,0]
        ]);
        $this->assertEquals($trues->toArray(),$b->toArray());

        // batchDims = 1
        // axis = 1
        // A:   2D [bathces,len]
        // X:   1D [bathces]
        // res: 3D [bathces,numClass,len]
        $shapeA = [4,2];
        $a = $this->array([
            [4,5],
            [10,11],
            [14,15],
            [18,19]
        ]);
        $x = $this->array([2,2,1,0],dtype:NDArray::int32);
        $shapeB = [4,3,2];
        $b = $this->zeros($shapeB);
        $matlib->gatherb(...$this->translate_scatterb($x,$a,$shapeB,axis:1,batchDims:1,outputs:$b));
        $this->assertEquals([4,2],$a->shape());
        $this->assertEquals([4],$x->shape());
        $this->assertEquals([4,3,2],$b->shape());
        $trues = $this->array([
            [[0,0],[0,0],[4,5]],
            [[0,0],[0,0],[10,11]],
            [[0,0],[14,15],[0,0]],
            [[18,19],[0,0],[0,0]]
        ]);
        $this->assertEquals($trues->toArray(),$b->toArray());

        // for argmax
        // batchDims = 1    // batchDims=axis
        // axis = 1         // default: axis=batchDims
        // detailDepth = 3  // detailDepth=params->ndim()
        // indexDepth = 1   // indexDepth=batchDims
        // A:   2D [batches,k]
        // X:   2D [batches,k]
        // res: 2D [batches,numClass,k]
        $shapeA = [3,4];
        $a = $this->array([
            [ 4, 5, 6, 7],
            [12,13,14,15],
            [20,21,22,23]
        ]);
        $x = $this->array([
            [1,1,1,1],
            [1,1,1,1],
            [1,1,1,1],
        ],dtype:NDArray::int32);
        $shapeB = [3,2,4];
        $b = $this->zeros($shapeB);
        $matlib->gatherb(...$this->translate_scatterb($x,$a,$shapeB,batchDims:1,detailDepth:count($shapeB),indexDepth:1,outputs:$b));
        $this->assertEquals([3,4],$a->shape());
        $this->assertEquals([3,4],$x->shape());
        $this->assertEquals([3,2,4],$b->shape());
        $trues = $this->array([
            [[0,0,0,0],[4,5,6,7]],
            [[0,0,0,0],[12,13,14,15]],
            [[0,0,0,0],[20,21,22,23]],
        ]);
        //echo $mo->toString($b,null,true)."\n";
        $this->assertEquals($trues->toArray(),$b->toArray());
        //$sourceDataShape = [3,2,4];
        //$sourceData = $this->array(range(0,array_product($sourceDataShape)-1))->reshape($sourceDataShape);
        //$max = $la->reduceMax($sourceData,axis:1);
        ////echo $mo->toString($max,null,true)."\n";
        //$imax = $la->reduceArgMax($sourceData,axis:1);
        ////echo $mo->toString($max,null,true)."\n";
        //$this->assertEquals($imax->toArray(),$x->toArray());
        //$d = $la->scatterb($imax,$max,$shapeB,batchDims:1,detailDepth:count($shapeB),indexDepth:1);
        ////echo $mo->toString($max,null,true)."\n";
        //$this->assertEquals($d->toArray(),$trues->toArray());

        // batchDims = 1
        // axis = 2
        // A:   2D [batches,m]
        // X:   1D [batches]
        // res: 3D [batches,m,numClass]
        $shapeA = [4,2,3];
        $a = $this->array([
            [ 2, 5],
            [ 8,11],
            [13,16],
            [18,21],
        ]);
        $x = $this->array([2,2,1,0],dtype:NDArray::int32);
        $shapeB = [4,2,3];
        $b = $this->zeros($shapeB);
        $matlib->gatherb(...$this->translate_scatterb($x,$a,$shapeB,axis:2,batchDims:1,outputs:$b));
        $this->assertEquals([4,2],$a->shape());
        $this->assertEquals([4],$x->shape());
        $this->assertEquals([4,2,3],$b->shape());
        $trues = $this->array([
            [[0,0,2],[0,0,5]],
            [[0,0,8],[0,0,11]],
            [[0,13,0],[0,16,0]],
            [[18,0,0],[21,0,0]],
        ]);
        $this->assertEquals($trues->toArray(),$b->toArray());

        // batchDims = 1
        // axis = 2
        // A:   3D [batches,m,len]
        // X:   1D [batches]
        // res: 4D [batches,m,numClass,len]
        $shapeA = [4,2,3,2];
        $a = $this->array([
            [[ 4, 5],[10,11]],
            [[16,17],[22,23]],
            [[26,27],[32,33]],
            [[36,37],[42,43]],
        ]);
        $x = $this->array([2,2,1,0],dtype:NDArray::int32);
        $shapeB = [4,2,3,2];
        $b = $this->zeros($shapeB);
        $matlib->gatherb(...$this->translate_scatterb($x,$a,$shapeB,axis:2,batchDims:1,outputs:$b));
        $this->assertEquals([4,2,2],$a->shape());
        $this->assertEquals([4],$x->shape());
        $this->assertEquals([4,2,3,2],$b->shape());
        $trues = $this->array([
            [[[0,0],[0,0],[4,5]],  [[0,0],[0,0],[10,11]]],
            [[[0,0],[0,0],[16,17]],[[0,0],[0,0],[22,23]]],
            [[[0,0],[26,27],[0,0]],[[0,0],[32,33],[0,0]]],
            [[[36,37],[0,0],[0,0]],[[42,43],[0,0],[0,0]]],
        ]);
        $this->assertEquals($trues->toArray(),$b->toArray());

        // batchDims = 1
        // axis = 1
        // A:   2D [batches,n]
        // X:   2D [batches,n]
        // res: 2D [batches,numClass]
        $shapeA = [4,2];
        $a = $this->array([
            [0,1],
            [4,3],
            [8,8],
            [10,9]
        ]);
        $x = $this->array([
            [0,1],
            [1,0],
            [2,2],
            [1,0]
        ],dtype:NDArray::int32);
        $shapeB = [4,3];
        $b = $this->zeros($shapeB);
        $matlib->gatherb(...$this->translate_scatterb($x,$a,$shapeB,axis:1,batchDims:1,outputs:$b));
        $this->assertEquals([4,2],$a->shape());
        $this->assertEquals([4,2],$x->shape());
        $this->assertEquals([4,3],$b->shape());
        $trues = $this->array([
            [0,1,0],
            [3,4,0],
            [0,0,8],
            [9,10,0]
        ]);
        $this->assertEquals($trues->toArray(),$b->toArray());

        // batchDims = 1
        // axis = 1
        // detailDepth = 3
        // indexDepth = 2
        // A:   3D [batches,n,k]
        // X:   3D [batches,n,k]
        // res: 3D [batches,numClass,k]
        $shapeA = [4,3,2];
        $a = $this->array([
            [[ 0, 3],[ 0, 3],[ 0, 3]],
            [[12,11],[12,11],[12,11]],
            [[24,25],[24,25],[24,25]],
            [[32,31],[32,31],[32,31]]
        ]);
        $x = $this->array([
            [[0,1],[0,1],[0,1]],
            [[1,0],[1,0],[1,0]],
            [[2,2],[2,2],[2,2]],
            [[1,0],[1,0],[1,0]]
        ],dtype:NDArray::int32);
        $shapeB = [4,5,2];
        $b = $this->zeros($shapeB);
        $matlib->gatherb(...$this->translate_scatterb($x,$a,$shapeB,axis:1,batchDims:1,detailDepth:3,indexDepth:2,outputs:$b));
        $this->assertEquals([4,3,2],$a->shape());
        $this->assertEquals([4,3,2],$x->shape());
        $this->assertEquals([4,5,2],$b->shape());
        $trues = $this->array([
            [[0,0], [0,3], [0,0],  [0,0],[0,0]],
            [[0,11],[12,0],[0,0],  [0,0],[0,0]],
            [[0,0], [0,0], [24,25],[0,0],[0,0]],
            [[0,31],[32,0],[0,0],  [0,0],[0,0]]
        ]);
        $this->assertEquals($trues->toArray(),$b->toArray());

        // batchDims = 1
        // axis = 1
        // detailDepth = 3
        // indexDepth = 2
        // res: 4D [batches,n,k,len]
        // X:   3D [batches,n,k]
        // A:   4D [batches,numClass,k,len]
        $shapeA = [4,3,2,3];
        $a = $this->array([
            [[[ 0, 1, 2],[ 9,10,11]],
             [[ 0, 1, 2],[ 9,10,11]],
             [[ 0, 1, 2],[ 9,10,11]],],
            [[[36,37,38],[33,34,35]],
             [[36,37,38],[33,34,35]],
             [[36,37,38],[33,34,35]],],
            [[[72,73,74],[75,76,77]],
             [[72,73,74],[75,76,77]],
             [[72,73,74],[75,76,77]],],
            [[[96,97,98],[93,94,95]],
             [[96,97,98],[93,94,95]],
             [[96,97,98],[93,94,95]],]
        ]);
        $x = $this->array([
            [[0,1],[0,1],[0,1]],
            [[1,0],[1,0],[1,0]],
            [[2,2],[2,2],[2,2]],
            [[1,0],[1,0],[1,0]],
        ],dtype:NDArray::int32);
        $shapeB = [4,5,2,3];
        $b = $this->zeros($shapeB);
        $matlib->gatherb(...$this->translate_scatterb($x,$a,$shapeB,axis:1,batchDims:1,detailDepth:3,indexDepth:2,outputs:$b));
        $this->assertEquals([4,3,2,3],$a->shape());
        $this->assertEquals([4,3,2],$x->shape());
        $this->assertEquals([4,5,2,3],$b->shape());
        $trues = $this->array([
            [[[0,1,2],   [0,0,0]],
             [[0,0,0],   [9,10,11]],
             [[0,0,0],   [0,0,0]],
             [[0,0,0],   [0,0,0]],
             [[0,0,0],   [0,0,0]]],
            [[[0,0,0],   [33,34,35]],
             [[36,37,38],[0,0,0]],
             [[0,0,0],   [0,0,0]],
             [[0,0,0],   [0,0,0]],
             [[0,0,0],   [0,0,0]]],
            [[[0,0,0],   [0,0,0]],
             [[0,0,0],   [0,0,0]],
             [[72,73,74],[75,76,77]],
             [[0,0,0],   [0,0,0]],
             [[0,0,0],   [0,0,0]]],
            [[[0,0,0],   [93,94,95]],
             [[96,97,98],[0,0,0]],
             [[0,0,0],   [0,0,0]],
             [[0,0,0],   [0,0,0]],
             [[0,0,0],   [0,0,0]]],
        ]);
        $this->assertEquals($trues->toArray(),$b->toArray());

        // batchDims = 1
        // axis = 2
        // detailDepth = 4
        // indexDepth = 2
        // A:   5D [batches,m,n,k,len]
        // X:   3D [batches,n,k]
        // res: 5D [batches,m,numClass,k,len]
        $shapeA = [4,2,3,2,3];
        $a = $this->array([
            [[[[ 0, 1, 2],[ 9,10,11]],
              [[ 0, 1, 2],[ 9,10,11]],
              [[ 0, 1, 2],[ 9,10,11]]],
             [[[30,31,32],[39,40,41]],
              [[30,31,32],[39,40,41]],
              [[30,31,32],[39,40,41]]]],
            [[[[66,67,68],[63,64,65]],
              [[66,67,68],[63,64,65]],
              [[66,67,68],[63,64,65]]],
             [[[96,97,98],[93,94,95]],
              [[96,97,98],[93,94,95]],
              [[96,97,98],[93,94,95]]]],
            [[[[132,133,134],[135,136,137]],
              [[132,133,134],[135,136,137]],
              [[132,133,134],[135,136,137]]],
             [[[162,163,164],[165,166,167]],
              [[162,163,164],[165,166,167]],
              [[162,163,164],[165,166,167]]]],
            [[[[186,187,188],[183,184,185]],
              [[186,187,188],[183,184,185]],
              [[186,187,188],[183,184,185]]],
             [[[216,217,218],[213,214,215]],
              [[216,217,218],[213,214,215]],
              [[216,217,218],[213,214,215]]]]
        ]);
        $x = $this->array([
            [[0,1],[0,1],[0,1]],
            [[1,0],[1,0],[1,0]],
            [[2,2],[2,2],[2,2]],
            [[1,0],[1,0],[1,0]],
        ],dtype:NDArray::int32);
        $shapeB = [4,2,5,2,3];
        $b = $this->zeros($shapeB);
        $matlib->gatherb(...$this->translate_scatterb($x,$a,$shapeB,axis:2,batchDims:1,detailDepth:4,indexDepth:2,outputs:$b));
        $this->assertEquals([4,2,3,2,3],$a->shape());
        $this->assertEquals([4,3,2],$x->shape());
        $this->assertEquals([4,2,5,2,3],$b->shape());
        $trues = $this->array([
            [[[[ 0, 1, 2],[ 0, 0, 0]],
              [[ 0, 0, 0],[ 9,10,11]],
              [[ 0, 0, 0],[ 0, 0, 0]],
              [[ 0, 0, 0],[ 0, 0, 0]],
              [[ 0, 0, 0],[ 0, 0, 0]]],
             [[[30,31,32],[ 0, 0, 0]],
              [[ 0, 0, 0],[39,40,41]],
              [[ 0, 0, 0],[ 0, 0, 0]],
              [[ 0, 0, 0],[ 0, 0, 0]],
              [[ 0, 0, 0],[ 0, 0, 0]]]],
            [[[[ 0, 0, 0],[63,64,65]],
              [[66,67,68],[ 0, 0, 0]],
              [[ 0, 0, 0],[ 0, 0, 0]],
              [[ 0, 0, 0],[ 0, 0, 0]],
              [[ 0, 0, 0],[ 0, 0, 0]]],
             [[[ 0, 0, 0],[93,94,95]],
              [[96,97,98],[ 0, 0, 0]],
              [[ 0, 0, 0],[ 0, 0, 0]],
              [[ 0, 0, 0],[ 0, 0, 0]],
              [[ 0, 0, 0],[ 0, 0, 0]]]],
            [[[[  0,  0,  0],[  0,  0,  0]],
              [[  0,  0,  0],[  0,  0,  0]],
              [[132,133,134],[135,136,137]],
              [[  0,  0,  0],[  0,  0,  0]],
              [[  0,  0,  0],[  0,  0,  0]]],
             [[[  0,  0,  0],[  0,  0,  0]],
              [[  0,  0,  0],[  0,  0,  0]],
              [[162,163,164],[165,166,167]],
              [[  0,  0,  0],[  0,  0,  0]],
              [[  0,  0,  0],[  0,  0,  0]]]],
            [[[[  0,  0,  0],[183,184,185]],
              [[186,187,188],[  0,  0,  0]],
              [[  0,  0,  0],[  0,  0,  0]],
              [[  0,  0,  0],[  0,  0,  0]],
              [[  0,  0,  0],[  0,  0,  0]]],
             [[[  0,  0,  0],[213,214,215]],
              [[216,217,218],[  0,  0,  0]],
              [[  0,  0,  0],[  0,  0,  0]],
              [[  0,  0,  0],[  0,  0,  0]],
              [[  0,  0,  0],[  0,  0,  0]]]]
        ]);
        $this->assertEquals($trues->toArray(),$b->toArray());
    }

    public function testScatterbAddNormal()
    {
        $matlib = $this->getMatlib();

        // for Embedding backword.
        // parallel on scatter add mode
        // axis = 0
        // A:   1D [n]
        // X:   1D [n]
        // res: 1D [numClass]
        $shapeA = [3];
        $a = $this->array([2,2,1]);
        $x = $this->array([2,2,1], dtype:NDArray::int32);
        $shapeB = [4];
        $b = $this->zeros($shapeB);
        $matlib->gatherb(...$this->translate_scatterbAdd($x, $a, $shapeB,outputs:$b));
        $this->assertEquals([3], $a->shape());
        $this->assertEquals([3], $x->shape());
        $this->assertEquals([4], $b->shape());
        $trues = $this->array([0,1,4,0]);
        $this->assertEquals($trues->toArray(),$b->toArray());

        // parallel on batch mode
        // batchDims = 1
        // axis = 1
        // A:   2D [batches]
        // X:   1D [batches]
        // res: 2D [batches,numClass]
        $shapeA = [4];
        $a = $this->array([2,5,7,9]);
        $x = $this->array([2,2,1,0],dtype:NDArray::int32);
        $shapeB = [4,3];
        $b = $this->zeros($shapeB);
        $matlib->gatherb(...$this->translate_scatterbAdd($x,$a,$shapeB,axis:1,batchDims:1,outputs:$b));
        $this->assertEquals([4],$a->shape());
        $this->assertEquals([4],$x->shape());
        $this->assertEquals([4,3],$b->shape());
        $trues = $this->array([
            [0,0,2],
            [0,0,5],
            [0,7,0],
            [9,0,0]
        ]);
        $this->assertEquals($trues->toArray(),$b->toArray());
    }

    public function testGatherNDNormal()
    {
        $matlib = $this->getMatlib();

        // axis = 0
        // 1D indices
        // A:   1D [p0]
        // X:   1D [depth]
        // res: 0D []
        $shapeA = [5];
        $a = $this->array(range(0,array_product($shapeA)-1))->reshape($shapeA);
        $x = $this->array(
            [1]
        ,dtype:NDArray::int32);
        $b = $this->zeros([]);
        $matlib->gathernd(...$this->translate_gatherND($a,$x,batchDims:0,outputs:$b));
        $this->assertEquals([5],$a->shape());
        $this->assertEquals([1],$x->shape());
        $this->assertEquals([],$b->shape());
        $this->assertEquals(
            1
        ,$b->toArray());

        // axis = 0
        // 1D indices
        // A:   1D [p0]
        // X:   2D [n,depth]
        // res: 1D [n]
        $shapeA = [5];
        $a = $this->array(range(0,array_product($shapeA)-1))->reshape($shapeA);
        $x = $this->array([
            [1],
            [2],
        ],dtype:NDArray::int32);
        $b = $this->zeros([2]);
        $matlib->gathernd(...$this->translate_gatherND($a,$x,batchDims:0,outputs:$b));
        $this->assertEquals([5],$a->shape());
        $this->assertEquals([2,1],$x->shape());
        $this->assertEquals([2],$b->shape());
        $this->assertEquals([
            1,
            2,
        ],$b->toArray());

        // axis = 0
        // 1D indices
        // A:   2D [p0,k]
        // X:   2D [n,depth]
        // res: 2D [n,k]
        $shapeA = [5,2];
        $a = $this->array(range(0,array_product($shapeA)-1))->reshape($shapeA);
        $x = $this->array([
            [1],
            [2],
        ],dtype:NDArray::int32);
        $b = $this->zeros([2,2]);
        $matlib->gathernd(...$this->translate_gatherND($a,$x,batchDims:0,outputs:$b));
        $this->assertEquals([5,2],$a->shape());
        $this->assertEquals([2,1],$x->shape());
        $this->assertEquals([2,2],$b->shape());
        $this->assertEquals([
            [2,3],
            [4,5],
        ],$b->toArray());

        // axis = 0
        // 2D indices
        // A:   3D [p0,p1,k]
        // X:   2D [n,depth]
        // res: 2D [n,k]
        $shapeA = [5,6,2];
        $a = $this->array(range(0,array_product($shapeA)-1))->reshape($shapeA);
        $x = $this->array([
            [0,1],
            [2,3],
        ],dtype:NDArray::int32);
        $b = $this->zeros([2,2]);
        $matlib->gathernd(...$this->translate_gatherND($a,$x,batchDims:0,outputs:$b));
        $this->assertEquals([5,6,2],$a->shape());
        $this->assertEquals([2,2],$x->shape());
        $this->assertEquals([2,2],$b->shape());
        $this->assertEquals([
            [2,3],
            [30,31],
        ],$b->toArray());


        // axis = 1
        // 1D indices
        // A:   2D [m,p0]
        // X:   2D [m,depth]
        // res: 1D [m]
        $shapeA = [4,5];
        $a = $this->array(range(0,array_product($shapeA)-1))->reshape($shapeA);
        $x = $this->array([
            [0],
            [1],
            [2],
            [0],
        ],dtype:NDArray::int32);
        $b = $this->zeros([4]);
        $matlib->gathernd(...$this->translate_gatherND($a,$x,batchDims:1,outputs:$b));
        $this->assertEquals([4,5],$a->shape());
        $this->assertEquals([4,1],$x->shape());
        $this->assertEquals([4],$b->shape());
        $this->assertEquals([
            0,
            6,
            12,
            15
        ],$b->toArray());

        // axis = 1
        // 1D indices
        // A:   3D [m,p0,k]
        // X:   2D [m,depth]
        // res: 2D [m,k]
        $shapeA = [4,5,2];
        $a = $this->array(range(0,array_product($shapeA)-1))->reshape($shapeA);
        $x = $this->array([
            [0],
            [1],
            [2],
            [0],
        ],dtype:NDArray::int32);
        $b = $this->zeros([4,2]);
        $matlib->gathernd(...$this->translate_gatherND($a,$x,batchDims:1,outputs:$b));
        $this->assertEquals([4,5,2],$a->shape());
        $this->assertEquals([4,1],$x->shape());
        $this->assertEquals([4,2],$b->shape());
        $this->assertEquals([
            [ 0,  1],
            [12, 13],
            [24, 25],
            [30, 31],
        ],$b->toArray());

        // axis = 1
        // 1D indices
        // A:   3D [m,p0,k]
        // X:   3D [m,n,depth]
        // res: 3D [m,n,k]
        $shapeA = [4,5,2];
        $a = $this->array(range(0,array_product($shapeA)-1))->reshape($shapeA);
        $x = $this->array([
            [[0],
             [0]],
            [[1],
             [1]],
            [[2],
             [2]],
            [[0],
             [0]],
        ],dtype:NDArray::int32);
        $b = $this->zeros([4,2,2]);
        $matlib->gathernd(...$this->translate_gatherND($a,$x,batchDims:1,outputs:$b));
        $this->assertEquals([4,5,2],$a->shape());
        $this->assertEquals([4,2,1],$x->shape());
        $this->assertEquals([4,2,2],$b->shape());
        $this->assertEquals([
           [[ 0,  1],
            [ 0,  1]],
           [[12, 13],
            [12, 13]],
           [[24, 25],
            [24, 25]],
           [[30, 31],
            [30, 31]],
        ],$b->toArray());

        // axis = 1
        // 2D indices
        // A:   4D [m,p0,p1,k]
        // X:   3D [m,n,depth]
        // res: 3D [m,n,k]
        $shapeA = [4,5,6,2];
        $a = $this->array(range(0,array_product($shapeA)-1))->reshape($shapeA);
        $x = $this->array([
            [[0,0],[0,1],[1,0]],
            [[0,1],[1,2],[4,5]],
            [[2,1],[0,1],[0,1]],
            [[1,1],[1,1],[1,1]],
        ],dtype:NDArray::int32);
        $b = $this->zeros([4,3,2]);
        $matlib->gathernd(...$this->translate_gatherND($a,$x,batchDims:1,outputs:$b));
        $this->assertEquals([4,5,6,2],$a->shape());
        $this->assertEquals([4,3,2],$x->shape());
        $this->assertEquals([4,3,2],$b->shape());
        $this->assertEquals([
           [[  0,   1],
            [  2,   3],
            [ 12,  13]],
           [[ 62,  63],
            [ 76,  77],
            [118, 119]],
           [[146, 147],
            [122, 123],
            [122, 123]],
           [[194, 195],
            [194, 195],
            [194, 195]],
        ],$b->toArray());
    }


    public function testScatterAxisNull()
    {
        $matlib = $this->getMatlib();
        // float32
        $numClass = 4;
        $X = $this->array([0,2],NDArray::int64);
        $A = $this->array([[1,2,3],[7,8,9]],NDArray::float32);
        $B = $this->zeros([4,3],$A->dtype());
        [$reduce,$reverse,$addMode,$n,$k,$numClass,$XX,$offX,$BB,$offB,$AA,$offA]
            = $this->translate_scatter($X,$A,$numClass,$axis=null,$B);
        $this->assertFalse($reduce);
        $matlib->gather($reverse,$addMode,$n,$k,$numClass,$XX,$offX,$BB,$offB,$AA,$offA);

        $this->assertEquals(
           [[1,2,3],
            [0,0,0],
            [7,8,9],
            [0,0,0]],
            $B->toArray()
        );

        // float64
        $X = $this->array([0,2],NDArray::int64);
        $A = $this->array([[1,2,3],[7,8,9]],NDArray::float64);
        $B = $this->zeros([4,3],$A->dtype());
        [$reduce,$reverse,$addMode,$n,$k,$numClass,$XX,$offX,$BB,$offB,$AA,$offA]
            = $this->translate_scatter($X,$A,$numClass,$axis=null,$B);
        $this->assertFalse($reduce);
        $matlib->gather($reverse,$addMode,$n,$k,$numClass,$XX,$offX,$BB,$offB,$AA,$offA);

        $this->assertEquals(
           [[1,2,3],
            [0,0,0],
            [7,8,9],
            [0,0,0]],
            $B->toArray()
        );
        // int64
        $X = $this->array([0,2],NDArray::int64);
        $A = $this->array([[1,2,3],[7,8,9]],NDArray::int64);
        $B = $this->zeros([4,3],$A->dtype());
        [$reduce,$reverse,$addMode,$n,$k,$numClass,$XX,$offX,$BB,$offB,$AA,$offA]
            = $this->translate_scatter($X,$A,$numClass,$axis=null,$B);
        $this->assertFalse($reduce);
        $matlib->gather($reverse,$addMode,$n,$k,$numClass,$XX,$offX,$BB,$offB,$AA,$offA);
        $this->assertEquals(
           [[1,2,3],
            [0,0,0],
            [7,8,9],
            [0,0,0]],
            $B->toArray()
        );
        // uint8
        $X = $this->array([0,2],NDArray::int64);
        $A = $this->array([[1,2,3],[7,8,9]],NDArray::uint8);
        $B = $this->zeros([4,3],$A->dtype());
        [$reduce,$reverse,$addMode,$n,$k,$numClass,$XX,$offX,$BB,$offB,$AA,$offA]
            = $this->translate_scatter($X,$A,$numClass,$axis=null,$B);
        $this->assertFalse($reduce);
        $matlib->gather($reverse,$addMode,$n,$k,$numClass,$XX,$offX,$BB,$offB,$AA,$offA);
        $this->assertEquals(
           [[1,2,3],
            [0,0,0],
            [7,8,9],
            [0,0,0]],
            $B->toArray()
        );
        // float32
        $X = $this->array([0,2],NDArray::int64);
        $A = $this->array([1,3],NDArray::float32);
        $B = $this->zeros([4],$A->dtype());
        [$reduce,$reverse,$addMode,$n,$k,$numClass,$XX,$offX,$BB,$offB,$AA,$offA]
            = $this->translate_scatter($X,$A,$numClass,$axis=null,$B);
        $this->assertFalse($reduce);
        $matlib->gather($reverse,$addMode,$n,$k,$numClass,$XX,$offX,$BB,$offB,$AA,$offA);
        $this->assertEquals(
           [1,0,3,0],
            $B->toArray()
        );
        // int32
        $X = $this->array([0,2],NDArray::int64);
        $A = $this->array([1,3],NDArray::int32);
        $B = $this->zeros([4],$A->dtype());
        [$reduce,$reverse,$addMode,$n,$k,$numClass,$XX,$offX,$BB,$offB,$AA,$offA]
            = $this->translate_scatter($X,$A,$numClass,$axis=null,$B);
        $this->assertFalse($reduce);
        $matlib->gather($reverse,$addMode,$n,$k,$numClass,$XX,$offX,$BB,$offB,$AA,$offA);
        $this->assertEquals(
           [1,0,3,0],
            $B->toArray()
        );
        // float64
        $X = $this->array([0,2],NDArray::int64);
        $A = $this->array([1,3],NDArray::float64);
        $B = $this->zeros([4],$A->dtype());
        [$reduce,$reverse,$addMode,$n,$k,$numClass,$XX,$offX,$BB,$offB,$AA,$offA]
            = $this->translate_scatter($X,$A,$numClass,$axis=null,$B);
        $this->assertFalse($reduce);
        $matlib->gather($reverse,$addMode,$n,$k,$numClass,$XX,$offX,$BB,$offB,$AA,$offA);
        $this->assertEquals(
           [1,0,3,0],
            $B->toArray()
        );
        // int64
        $X = $this->array([0,2],NDArray::int64);
        $A = $this->array([1,3],NDArray::int64);
        $B = $this->zeros([4],$A->dtype());
        [$reduce,$reverse,$addMode,$n,$k,$numClass,$XX,$offX,$BB,$offB,$AA,$offA]
            = $this->translate_scatter($X,$A,$numClass,$axis=null,$B);
        $this->assertFalse($reduce);
        $matlib->gather($reverse,$addMode,$n,$k,$numClass,$XX,$offX,$BB,$offB,$AA,$offA);
        $this->assertEquals(
           [1,0,3,0],
            $B->toArray()
        );
        // uint8
        $X = $this->array([0,2],NDArray::int64);
        $A = $this->array([252,254],NDArray::uint8);
        $B = $this->zeros([4],$A->dtype());
        [$reduce,$reverse,$addMode,$n,$k,$numClass,$XX,$offX,$BB,$offB,$AA,$offA]
            = $this->translate_scatter($X,$A,$numClass,$axis=null,$B);
        $this->assertFalse($reduce);
        $matlib->gather($reverse,$addMode,$n,$k,$numClass,$XX,$offX,$BB,$offB,$AA,$offA);
        $this->assertEquals(
           [252,0,254,0],
            $B->toArray()
        );
        // x=uint8
        $X = $this->array([0,255],NDArray::uint8);
        $A = $this->array([252,254],NDArray::uint8);
        $B = $this->zeros([256],$A->dtype());
        $numClass = 256;
        [$reduce,$reverse,$addMode,$n,$k,$numClass,$XX,$offX,$BB,$offB,$AA,$offA]
            = $this->translate_scatter($X,$A,$numClass,$axis=null,$B);
        $this->assertFalse($reduce);
        $matlib->gather($reverse,$addMode,$n,$k,$numClass,$XX,$offX,$BB,$offB,$AA,$offA);
        $this->assertEquals(252,$B[0]);
        $this->assertEquals(254,$B[255]);
    }

    public function testScatterAxis1()
    {
        $matlib = $this->getMatlib();
        $numClass = 3;
        $X = $this->array([0,1,2,0],NDArray::int32);
        $A = $this->array([1,5,9,10],NDArray::float32);
        $B = $this->zeros([4,3],$A->dtype());
        [$reduce,$reverse,$addMode,$m,$n,$numClass,$XX,$offX,$BB,$offB,$AA,$offA]
            = $this->translate_scatter($X,$A,$numClass,$axis=1,$B);
        $this->assertTrue($reduce);
        $matlib->reduceGather($reverse,$addMode,$m,$n,$numClass,$XX,$offX,$BB,$offB,$AA,$offA);

        $this->assertEquals(
           [[1,0,0],
            [0,5,0],
            [0,0,9],
            [10,0,0]],
            $B->toArray());

        $X = $this->array([0,1,2,0],NDArray::int64);
        $B = $this->zeros([4,3],$A->dtype());
        [$reduce,$reverse,$addMode,$m,$n,$numClass,$XX,$offX,$BB,$offB,$AA,$offA]
            = $this->translate_scatter($X,$A,$numClass,$axis=1,$B);
        $this->assertTrue($reduce);
        $matlib->reduceGather($reverse,$addMode,$m,$n,$numClass,$XX,$offX,$BB,$offB,$AA,$offA);

        $this->assertEquals(
           [[1,0,0],
            [0,5,0],
            [0,0,9],
            [10,0,0]],
            $B->toArray());

        $X = $this->array([0,1,2,0],NDArray::float32);
        $B = $this->zeros([4,3],$A->dtype());
        [$reduce,$reverse,$addMode,$m,$n,$numClass,$XX,$offX,$BB,$offB,$AA,$offA]
            = $this->translate_scatter($X,$A,$numClass,$axis=1,$B);
        $this->assertTrue($reduce);
        $matlib->reduceGather($reverse,$addMode,$m,$n,$numClass,$XX,$offX,$BB,$offB,$AA,$offA);
        $this->assertEquals(
           [[1,0,0],
            [0,5,0],
            [0,0,9],
            [10,0,0]],
            $B->toArray());

        $X = $this->array([0,1,2,0],NDArray::float64);
        $B = $this->zeros([4,3],$A->dtype());
        [$reduce,$reverse,$addMode,$m,$n,$numClass,$XX,$offX,$BB,$offB,$AA,$offA]
            = $this->translate_scatter($X,$A,$numClass,$axis=1,$B);
        $this->assertTrue($reduce);
        $matlib->reduceGather($reverse,$addMode,$m,$n,$numClass,$XX,$offX,$BB,$offB,$AA,$offA);
        $this->assertEquals(
           [[1,0,0],
            [0,5,0],
            [0,0,9],
            [10,0,0]],
            $B->toArray());
    }

    public function testScatterNDNormal()
    {
        $matlib = $this->getMatlib();

        // axis = 0
        // 1D indices
        // A:   0D []
        // X:   1D [depth]
        // res: 1D [p0]
        $a = $this->array(1);
        $x = $this->array(
            [1]
        ,dtype:NDArray::int32);
        $shape = [5];
        $b = $this->zeros($shape);
        $matlib->gathernd(...$this->translate_scatterND($x,$a,shape:$shape,batchDims:0,outputs:$b));
        $this->assertEquals([],$a->shape());
        $this->assertEquals([1],$x->shape());
        $this->assertEquals([5],$b->shape());
        $this->assertEquals(
            [0,1,0,0,0]
        ,$b->toArray());

        // axis = 0
        // 1D indices
        // A:   1D [n]
        // X:   2D [n,depth]
        // res: 1D [p0]
        $shapeA = [2];
        $a = $this->array([
            1,
            2,
        ]);
        $x = $this->array([
            [1],
            [2],
        ],dtype:NDArray::int32);
        $shape = [5];
        $b = $this->zeros($shape);
        $matlib->gathernd(...$this->translate_scatterND($x,$a,shape:$shape,batchDims:0,outputs:$b));
        $this->assertEquals([2],$a->shape());
        $this->assertEquals([2,1],$x->shape());
        $this->assertEquals([5],$b->shape());
        $this->assertEquals([
            0,
            1,
            2,
            0,
            0,
        ],$b->toArray());

        // axis = 0
        // 1D indices
        // A:   2D [n,k]
        // X:   2D [n,depth]
        // res: 2D [p0,k]
        $a = $this->array([
            [2,3],
            [4,5],
        ]);
        $x = $this->array([
            [1],
            [2],
        ],dtype:NDArray::int32);
        $shape = [5,2];
        $b = $this->zeros($shape);
        $matlib->gathernd(...$this->translate_scatterND($x,$a,shape:$shape,batchDims:0,outputs:$b));
        $this->assertEquals([2,2],$a->shape());
        $this->assertEquals([2,1],$x->shape());
        $this->assertEquals([5,2],$b->shape());
        $this->assertEquals([
            [0,0],
            [2,3],
            [4,5],
            [0,0],
            [0,0],
        ],$b->toArray());

        // axis = 0
        // 2D indices
        // A:   2D [n,k]
        // X:   2D [n,depth]
        // res: 3D [p0,p1,k]
        $a = $this->array([
            [2,3],
            [30,31],
        ]);
        $x = $this->array([
            [0,1],
            [2,3],
        ],dtype:NDArray::int32);
        $shape = [5,6,2];
        $b = $this->zeros($shape);
        $matlib->gathernd(...$this->translate_scatterND($x,$a,shape:$shape,batchDims:0,outputs:$b));
        $this->assertEquals([2,2],$a->shape());
        $this->assertEquals([2,2],$x->shape());
        $this->assertEquals([5,6,2],$b->shape());
        $this->assertEquals([
            [[0,0],[2,3],[0,0],[0,0],  [0,0],[0,0]],
            [[0,0],[0,0],[0,0],[0,0],  [0,0],[0,0]],
            [[0,0],[0,0],[0,0],[30,31],[0,0],[0,0]],
            [[0,0],[0,0],[0,0],[0,0],  [0,0],[0,0]],
            [[0,0],[0,0],[0,0],[0,0],  [0,0],[0,0]],
        ],$b->toArray());


        // axis = 1
        // 1D indices
        // A:   1D [m]
        // X:   2D [m,depth]
        // res: 2D [m,p0]
        $a = $this->array([
            0,
            6,
            12,
            15
        ]);
        $x = $this->array([
            [0],
            [1],
            [2],
            [0],
        ],dtype:NDArray::int32);
        $shape = [4,5];
        $b = $this->zeros($shape);
        $matlib->gathernd(...$this->translate_scatterND($x,$a,shape:$shape,batchDims:1,outputs:$b));
        $this->assertEquals([4],$a->shape());
        $this->assertEquals([4,1],$x->shape());
        $this->assertEquals([4,5],$b->shape());
        $this->assertEquals([
            [ 0, 0, 0, 0, 0],
            [ 0, 6, 0, 0, 0],
            [ 0, 0,12, 0, 0],
            [15, 0, 0, 0, 0],
        ],$b->toArray());

        // axis = 1
        // 1D indices
        // A:   2D [m,k]
        // X:   2D [m,depth]
        // res: 3D [m,p0,k]
        $a = $this->array([
            [ 0,  1],
            [12, 13],
            [24, 25],
            [30, 31],
        ]);
        $x = $this->array([
            [0],
            [1],
            [2],
            [0],
        ],dtype:NDArray::int32);
        $shape = [4,5,2];
        $b = $this->zeros($shape);
        $matlib->gathernd(...$this->translate_scatterND($x,$a,shape:$shape,batchDims:1,outputs:$b));
        $this->assertEquals([4,2],$a->shape());
        $this->assertEquals([4,1],$x->shape());
        $this->assertEquals([4,5,2],$b->shape());
        $this->assertEquals([
            [[0,1],[0,0],[0,0],[0,0],[0,0]],
            [[0,0],[12,13],[0,0],[0,0],[0,0]],
            [[0,0],[0,0],[24,25],[0,0],[0,0]],
            [[30,31],[0,0],[0,0],[0,0],[0,0]],
        ],$b->toArray());

        // axis = 1
        // 1D indices
        // A:   3D [m,n,k]
        // X:   3D [m,n,depth]
        // res: 3D [m,p0,k]
        $a = $this->array([
            [[ 0,  1],
            [ 0,  1]],
           [[12, 13],
            [12, 13]],
           [[24, 25],
            [24, 25]],
           [[30, 31],
            [30, 31]],
        ]);
        $x = $this->array([
            [[0],
             [0]],
            [[1],
             [1]],
            [[2],
             [2]],
            [[0],
             [0]],
        ],dtype:NDArray::int32);
        $shape = [4,5,2];
        $b = $this->zeros($shape);
        $matlib->gathernd(...$this->translate_scatterND($x,$a,shape:$shape,batchDims:1,outputs:$b));
        $this->assertEquals([4,2,2],$a->shape());
        $this->assertEquals([4,2,1],$x->shape());
        $this->assertEquals([4,5,2],$b->shape());
        $this->assertEquals([
            [[0,1],[0,0],[0,0],[0,0],[0,0]],
            [[0,0],[12,13],[0,0],[0,0],[0,0]],
            [[0,0],[0,0],[24,25],[0,0],[0,0]],
            [[30,31],[0,0],[0,0],[0,0],[0,0]],
        ],$b->toArray());

        // axis = 1
        // 2D indices
        // A:   3D [m,n,k]
        // X:   3D [m,n,depth]
        // res: 4D [m,p0,p1,k]
        $a = $this->array([
            [[  0,   1],
            [  2,   3],
            [ 12,  13]],
           [[ 62,  63],
            [ 76,  77],
            [118, 119]],
           [[146, 147],
            [122, 123],
            [122, 123]],
           [[194, 195],
            [194, 195],
            [194, 195]],
        ]);
        $x = $this->array([
            [[0,0],[0,1],[1,0]],
            [[0,1],[1,2],[4,5]],
            [[2,1],[0,1],[0,1]],
            [[1,1],[1,1],[1,1]],
        ],dtype:NDArray::int32);
        $shape = [4,5,6,2];
        $b = $this->zeros($shape);
        $matlib->gathernd(...$this->translate_scatterND($x,$a,shape:$shape,batchDims:1,outputs:$b));
        $this->assertEquals([4,3,2],$a->shape());
        $this->assertEquals([4,3,2],$x->shape());
        $this->assertEquals([4,5,6,2],$b->shape());
        $this->assertEquals([
            [[[0,1],[2,3],[0,0],[0,0],[0,0],[0,0]],
             [[12,13],[0,0],[0,0],[0,0],[0,0],[0,0]],
             [[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]],
             [[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]],
             [[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]]],
            [[[0,0],[62,63],[0,0],[0,0],[0,0],[0,0]],
             [[0,0],[0,0],[76,77],[0,0],[0,0],[0,0]],
             [[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]],
             [[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]],
             [[0,0],[0,0],[0,0],[0,0],[0,0],[118,119]]],
            [[[0,0],[122,123],[0,0],[0,0],[0,0],[0,0]],
             [[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]],
             [[0,0],[146,147],[0,0],[0,0],[0,0],[0,0]],
             [[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]],
             [[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]]],
            [[[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]],
             [[0,0],[194,195],[0,0],[0,0],[0,0],[0,0]],
             [[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]],
             [[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]],
             [[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]]],
        ],$b->toArray());
    }

    public function testScatterNDAddNormal()
    {
        $matlib = $this->getMatlib();

        // axis = 0
        // 2D indices
        // A:   2D [n,k]
        // X:   2D [n,depth]
        // res: 3D [p0,p1,k]
        $a = $this->array([
            [1,10],
            [2,3],
            [30,31],
            [1,10],
        ]);
        $x = $this->array([
            [4,5],
            [0,1],
            [2,3],
            [4,5],
        ],dtype:NDArray::int32);
        $shape = [5,6,2];
        $b = $this->zeros($shape);
        $matlib->gathernd(...$this->translate_scatterNDAdd($x,$a,shape:$shape,batchDims:0,outputs:$b));
        $this->assertEquals([4,2],$a->shape());
        $this->assertEquals([4,2],$x->shape());
        $this->assertEquals([5,6,2],$b->shape());
        $this->assertEquals([
            [[0,0],[2,3],[0,0],[0,0],  [0,0],[0,0]],
            [[0,0],[0,0],[0,0],[0,0],  [0,0],[0,0]],
            [[0,0],[0,0],[0,0],[30,31],[0,0],[0,0]],
            [[0,0],[0,0],[0,0],[0,0],  [0,0],[0,0]],
            [[0,0],[0,0],[0,0],[0,0],  [0,0],[2,20]],
        ],$b->toArray());

    }

    #[DataProvider('providerDtypesFloatsAndInteger8')]
    public function testsliceNormal($params)
    {
        extract($params);
        $matlib = $this->getMatlib();
        // 3D
        $x = $this->array([
            [[0,1,2],
             [3,4,5],
             [6,7,8],
             [9,10,11]],
            [[12,13,14],
             [15,16,17],
             [18,19,20],
             [21,22,23]],
        ],dtype:$dtype);
        $this->assertEquals(3,$x->ndim());
        $y = $this->zeros([2,2,3],dtype:$dtype);
        [
            $reverse,
            $addMode,
            $m,
            $n,
            $k,
            $itemSize,
            $AA,$offsetA,$incA,
            $YY,$offsetY,$incY,
            $startAxis0,$sizeAxis0,
            $startAxis1,$sizeAxis1,
            $startAxis2,$sizeAxis2
        ] = $this->translate_slice(false,
            $x,
            $start=[0,1],
            $size=[-1,2],
            $y
            );

        $matlib->slice(
            $reverse,
            $addMode,
            $m,
            $n,
            $k,
            $itemSize,
            $AA,$offsetA,$incA,
            $YY,$offsetY,$incY,
            $startAxis0,$sizeAxis0,
            $startAxis1,$sizeAxis1,
            $startAxis2,$sizeAxis2
        );

        $this->assertEquals([
            [[3,4,5],
             [6,7,8],],
            [[15,16,17],
             [18,19,20],],
        ],$y->toArray());
    }

    public function testSliceBoolNormal()
    {
        $dtype = NDArray::bool;
        $matlib = $this->getMatlib();
        // 3D
        $x = $this->array([
            [[true, false,false],
             [true, true, false],
             [true, true, true],
             [false,true, true]],
            [[false,false,true],
             [false,true, true],
             [true, true, true],
             [true, true, false]],
        ],dtype:$dtype);
        $this->assertEquals(3,$x->ndim());
        $y = $this->ones([2,2,3],dtype:$dtype);
        [
            $reverse,
            $addMode,
            $m,
            $n,
            $k,
            $itemSize,
            $AA,$offsetA,$incA,
            $YY,$offsetY,$incY,
            $startAxis0,$sizeAxis0,
            $startAxis1,$sizeAxis1,
            $startAxis2,$sizeAxis2
        ] = $this->translate_slice(false,
            $x,
            $start=[0,1],
            $size=[-1,2],
            $y
            );

        $matlib->slice(
            $reverse,
            $addMode,
            $m,
            $n,
            $k,
            $itemSize,
            $AA,$offsetA,$incA,
            $YY,$offsetY,$incY,
            $startAxis0,$sizeAxis0,
            $startAxis1,$sizeAxis1,
            $startAxis2,$sizeAxis2
        );

        $this->assertEquals([
            [[true, true, false],
             [true, true, true ]],
            [[false,true, true ],
             [true, true, true]],
        ],$y->toArray());
    }

    #[DataProvider('providerDtypesFloats')]
    public function testRepeatNormal($params)
    {
        extract($params);
        $matlib = $this->getMatlib();

        // Y := X (duplicate 2 times)
        $X = $this->array([
            [1,2,3],
            [4,5,6]
        ],dtype:$dtype);
        $Y = $this->zeros([2,2,3],dtype:$dtype);

        [
            $m,
            $k,
            $repeats,
            $AA,$offA,
            $BB,$offB
        ] = $this->translate_repeat($X,repeats:2,axis:1,output:$Y);

        $matlib->repeat(
            $m,
            $k,
            $repeats,
            $AA,$offA,
            $BB,$offB
        );

        $this->assertEquals([2,3],$X->shape());
        $this->assertEquals([2,2,3],$Y->shape());
        $this->assertEquals([
            [1,2,3],
            [4,5,6]
        ],$X->toArray());
        $this->assertEquals([
            [[1,2,3],[1,2,3]],
            [[4,5,6],[4,5,6]],
        ],$Y->toArray());
    }

    #[DataProvider('providerDtypesFloats')]
    public function testupdateAddOnehotNormal($params)
    {
        extract($params);
        $matlib = $this->getMatlib();
        $X = $this->array([1, 2],dtype:NDArray::int32);
        $Y = $this->array([[10,10,10],[10,10,10]],dtype:$dtype);
        $numClass = 3;
        [$m,$n,$a,$XX,$offX,$incX,$YY,$offY,$ldY] = $this->translate_onehot(
            $X,$numClass,-1,$Y);

        $matlib->updateAddOnehot($m,$n,$a,$XX,$offX,$incX,$YY,$offY,$ldY);
        $this->assertEquals([[10,9,10],[10,10,9]],$Y->toArray());
    }

    public function testupdateAddOnehotOutOfboundsLabelNumber1()
    {
        $matlib = $this->getMatlib();
        $X = $this->array([1, 3],dtype:NDArray::int32);
        $Y = $this->array([[10,10,10],[10,10,10]]);
        $numClass = 3;
        [$m,$n,$a,$XX,$offX,$incX,$YY,$offY,$ldY] = $this->translate_onehot(
            $X,$numClass,-1,$Y);

        $this->expectException(RuntimeException::class);
        $this->expectExceptionMessage('Label number is out of bounds.');
        $matlib->updateAddOnehot($m,$n,$a,$XX,$offX,$incX,$YY,$offY,$ldY);
    }

    public function testupdateAddOnehotOutOfboundsLabelNumber2()
    {
        $matlib = $this->getMatlib();
        $X = $this->array([1, -1],dtype:NDArray::int32);
        $Y = $this->array([[10,10,10],[10,10,10]]);
        $numClass = 3;
        [$m,$n,$a,$XX,$offX,$incX,$YY,$offY,$ldY] = $this->translate_onehot(
            $X,$numClass,-1,$Y);

        $this->expectException(RuntimeException::class);
        $this->expectExceptionMessage('Label number is out of bounds.');
        $matlib->updateAddOnehot($m,$n,$a,$XX,$offX,$incX,$YY,$offY,$ldY);
    }

    public function testupdateAddOnehotMinusM()
    {
        $matlib = $this->getMatlib();
        $X = $this->array([1, -1],dtype:NDArray::int32);
        $Y = $this->array([[10,10,10],[10,10,10]]);
        $numClass = 3;
        [$m,$n,$a,$XX,$offX,$incX,$YY,$offY,$ldY] = $this->translate_onehot(
            $X,$numClass,-1,$Y);

        $m = 0;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument m must be greater than 0.');
        $matlib->updateAddOnehot($m,$n,$a,$XX,$offX,$incX,$YY,$offY,$ldY);
    }

    public function testupdateAddOnehotMinusN()
    {
        $matlib = $this->getMatlib();
        $X = $this->array([1, -1],dtype:NDArray::int32);
        $Y = $this->array([[10,10,10],[10,10,10]]);
        $numClass = 3;
        [$m,$n,$a,$XX,$offX,$incX,$YY,$offY,$ldY] = $this->translate_onehot(
            $X,$numClass,-1,$Y);

        $n = 0;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument n must be greater than 0.');
        $matlib->updateAddOnehot($m,$n,$a,$XX,$offX,$incX,$YY,$offY,$ldY);
    }

    public function testupdateAddOnehotMinusOffsetX()
    {
        $matlib = $this->getMatlib();
        $X = $this->array([1, -1],dtype:NDArray::int32);
        $Y = $this->array([[10,10,10],[10,10,10]]);
        $numClass = 3;
        [$m,$n,$a,$XX,$offX,$incX,$YY,$offY,$ldY] = $this->translate_onehot(
            $X,$numClass,-1,$Y);

        $offX = -1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument offsetX must be greater than or equals 0.');
        $matlib->updateAddOnehot($m,$n,$a,$XX,$offX,$incX,$YY,$offY,$ldY);
    }

    public function testupdateAddOnehotMinusIncX()
    {
        $matlib = $this->getMatlib();
        $X = $this->array([1, -1],dtype:NDArray::int32);
        $Y = $this->array([[10,10,10],[10,10,10]]);
        $numClass = 3;
        [$m,$n,$a,$XX,$offX,$incX,$YY,$offY,$ldY] = $this->translate_onehot(
            $X,$numClass,-1,$Y);

        $incX = 0;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument incX must be greater than 0.');
        $matlib->updateAddOnehot($m,$n,$a,$XX,$offX,$incX,$YY,$offY,$ldY);
    }

    public function testupdateAddOnehotIllegalBufferX()
    {
        $matlib = $this->getMatlib();
        $X = $this->array([1, -1],dtype:NDArray::int32);
        $Y = $this->array([[10,10,10],[10,10,10]]);
        $numClass = 3;
        [$m,$n,$a,$XX,$offX,$incX,$YY,$offY,$ldY] = $this->translate_onehot(
            $X,$numClass,-1,$Y);

        $XX = new \stdClass();
        $this->expectException(TypeError::class);
        $this->expectExceptionMessage('must be of type Interop\Polite\Math\Matrix\LinearBuffer');
        $matlib->updateAddOnehot($m,$n,$a,$XX,$offX,$incX,$YY,$offY,$ldY);
    }

    public function testupdateAddOnehotOverflowBufferXwithSize()
    {
        $matlib = $this->getMatlib();
        $X = $this->array([1, -1],dtype:NDArray::int32);
        $Y = $this->array([[10,10,10],[10,10,10]]);
        $numClass = 3;
        [$m,$n,$a,$XX,$offX,$incX,$YY,$offY,$ldY] = $this->translate_onehot(
            $X,$numClass,-1,$Y);

        $XX = $this->array([1])->buffer();
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Vector specification too large for bufferX');
        $matlib->updateAddOnehot($m,$n,$a,$XX,$offX,$incX,$YY,$offY,$ldY);
    }

    public function testupdateAddOnehotOverflowBufferXwithOffsetX()
    {
        $matlib = $this->getMatlib();
        $X = $this->array([1, -1],dtype:NDArray::int32);
        $Y = $this->array([[10,10,10],[10,10,10]]);
        $numClass = 3;
        [$m,$n,$a,$XX,$offX,$incX,$YY,$offY,$ldY] = $this->translate_onehot(
            $X,$numClass,-1,$Y);

        $offX = 1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Vector specification too large for bufferX');
        $matlib->updateAddOnehot($m,$n,$a,$XX,$offX,$incX,$YY,$offY,$ldY);
    }

    public function testupdateAddOnehotOverflowBufferXwithIncX()
    {
        $matlib = $this->getMatlib();
        $X = $this->array([1, -1],dtype:NDArray::int32);
        $Y = $this->array([[10,10,10],[10,10,10]]);
        $numClass = 3;
        [$m,$n,$a,$XX,$offX,$incX,$YY,$offY,$ldY] = $this->translate_onehot(
            $X,$numClass,-1,$Y);

        $incX = 2;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Vector specification too large for bufferX');
        $matlib->updateAddOnehot($m,$n,$a,$XX,$offX,$incX,$YY,$offY,$ldY);
    }

    public function testupdateAddOnehotMinusOffsetY()
    {
        $matlib = $this->getMatlib();
        $X = $this->array([1, -1],dtype:NDArray::int32);
        $Y = $this->array([[10,10,10],[10,10,10]]);
        $numClass = 3;
        [$m,$n,$a,$XX,$offX,$incX,$YY,$offY,$ldY] = $this->translate_onehot(
            $X,$numClass,-1,$Y);

        $offY = -1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument offsetY must be greater than or equals 0.');
        $matlib->updateAddOnehot($m,$n,$a,$XX,$offX,$incX,$YY,$offY,$ldY);
    }

    public function testupdateAddOnehotMinusLdY()
    {
        $matlib = $this->getMatlib();
        $X = $this->array([1, -1],dtype:NDArray::int32);
        $Y = $this->array([[10,10,10],[10,10,10]]);
        $numClass = 3;
        [$m,$n,$a,$XX,$offX,$incX,$YY,$offY,$ldY] = $this->translate_onehot(
            $X,$numClass,-1,$Y);

        $ldY = 0;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument ldY must be greater than 0.');
        $matlib->updateAddOnehot($m,$n,$a,$XX,$offX,$incX,$YY,$offY,$ldY);
    }

    public function testupdateAddOnehotIllegalBufferY()
    {
        $matlib = $this->getMatlib();
        $X = $this->array([1, -1],dtype:NDArray::int32);
        $Y = $this->array([[10,10,10],[10,10,10]]);
        $numClass = 3;
        [$m,$n,$a,$XX,$offX,$incX,$YY,$offY,$ldY] = $this->translate_onehot(
            $X,$numClass,-1,$Y);

        $YY = new \stdClass();
        $this->expectException(TypeError::class);
        $this->expectExceptionMessage('must be of type Interop\Polite\Math\Matrix\LinearBuffer');
        $matlib->updateAddOnehot($m,$n,$a,$XX,$offX,$incX,$YY,$offY,$ldY);
    }

    public function testupdateAddOnehotOverflowBufferYwithSize()
    {
        $matlib = $this->getMatlib();
        $X = $this->array([1, -1],dtype:NDArray::int32);
        $Y = $this->array([[10,10,10],[10,10,10]]);
        $numClass = 3;
        [$m,$n,$a,$XX,$offX,$incX,$YY,$offY,$ldY] = $this->translate_onehot(
            $X,$numClass,-1,$Y);

        $YY = $this->array([1])->buffer();
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Matrix specification too large for bufferY');
        $matlib->updateAddOnehot($m,$n,$a,$XX,$offX,$incX,$YY,$offY,$ldY);
    }

    public function testupdateAddOnehotOverflowBufferYwithOffsetY()
    {
        $matlib = $this->getMatlib();
        $X = $this->array([1, -1],dtype:NDArray::int32);
        $Y = $this->array([[10,10,10],[10,10,10]]);
        $numClass = 3;
        [$m,$n,$a,$XX,$offX,$incX,$YY,$offY,$ldY] = $this->translate_onehot(
            $X,$numClass,-1,$Y);

        $offY = 1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Matrix specification too large for bufferY');
        $matlib->updateAddOnehot($m,$n,$a,$XX,$offX,$incX,$YY,$offY,$ldY);
    }

    public function testupdateAddOnehotOverflowBufferYwithIncY()
    {
        $matlib = $this->getMatlib();
        $X = $this->array([1, -1],dtype:NDArray::int32);
        $Y = $this->array([[10,10,10],[10,10,10]]);
        $numClass = 3;
        [$m,$n,$a,$XX,$offX,$incX,$YY,$offY,$ldY] = $this->translate_onehot(
            $X,$numClass,-1,$Y);

        $ldY = 4;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Matrix specification too large for bufferY');
        $matlib->updateAddOnehot($m,$n,$a,$XX,$offX,$incX,$YY,$offY,$ldY);
    }

    #[DataProvider('providerDtypesFloats')]
    public function testreduceSumSameSizeNormal($params)
    {
        extract($params);
        if($this->checkSkip('reduceSum')){return;}

        $matlib = $this->getMatlib();

        $A = $this->array([[1,2,3],[4,5,6]],dtype:$dtype);
        $X = $this->array([0,0],dtype:$dtype);
        [$m,$n,$k,$AA,$offA,$BB,$offB] =
            $this->translate_reduceSum($A,$axis=1,$X);

        $matlib->reduceSum($m,$n,$k,$AA,$offA,$BB,$offB);
        $this->assertEquals([6,15],$X->toArray());
    }

    #[DataProvider('providerDtypesFloats')]
    public function testreduceSumBroadcastTranspose($params)
    {
        extract($params);
        if($this->checkSkip('reduceSum')){return;}

        $matlib = $this->getMatlib();

        $A = $this->array([[1,2,3],[4,5,6]],dtype:$dtype);
        $X = $this->array([0,0,0],dtype:$dtype);
        [$m,$n,$k,$AA,$offA,$BB,$offB] =
            $this->translate_reduceSum($A,$axis=0,$X);

        $matlib->reduceSum($m,$n,$k,$AA,$offA,$BB,$offB);
        $this->assertEquals([5,7,9],$X->toArray());
    }

    public function testreduceSumMinusM()
    {
        if($this->checkSkip('reduceSum')){return;}

        $matlib = $this->getMatlib();

        $X = $this->array([0,0]);
        $A = $this->array([[1,2,3],[4,5,6]]);
        [$m,$n,$k,$AA,$offA,$BB,$offB] =
            $this->translate_reduceSum($A,$axis=1,$X);

        $m = 0;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument m must be greater than 0.');
        $matlib->reduceSum($m,$n,$k,$AA,$offA,$BB,$offB);
    }

    public function testreduceSumMinusK()
    {
        if($this->checkSkip('reduceSum')){return;}

        $matlib = $this->getMatlib();

        $X = $this->array([0,0]);
        $A = $this->array([[1,2,3],[4,5,6]]);
        [$m,$n,$k,$AA,$offA,$BB,$offB] =
            $this->translate_reduceSum($A,$axis=1,$X);

        $k = 0;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument k must be greater than 0.');
        $matlib->reduceSum($m,$n,$k,$AA,$offA,$BB,$offB);
    }

    public function testreduceSumMinusN()
    {
        if($this->checkSkip('reduceSum')){return;}

        $matlib = $this->getMatlib();

        $X = $this->array([0,0]);
        $A = $this->array([[1,2,3],[4,5,6]]);
        [$m,$n,$k,$AA,$offA,$BB,$offB] =
            $this->translate_reduceSum($A,$axis=1,$X);

        $n = 0;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument n must be greater than 0.');
        $matlib->reduceSum($m,$n,$k,$AA,$offA,$BB,$offB);
    }

    public function testreduceSumMinusOffsetB()
    {
        if($this->checkSkip('reduceSum')){return;}

        $matlib = $this->getMatlib();

        $X = $this->array([0,0]);
        $A = $this->array([[1,2,3],[4,5,6]]);
        [$m,$n,$k,$AA,$offA,$BB,$offB] =
            $this->translate_reduceSum($A,$axis=1,$X);

        $offB = -1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument offsetB must be greater than or equals 0.');
        $matlib->reduceSum($m,$n,$k,$AA,$offA,$BB,$offB);
    }

    public function testreduceSumIllegalBufferB()
    {
        if($this->checkSkip('reduceSum')){return;}

        $matlib = $this->getMatlib();

        $X = $this->array([0,0]);
        $A = $this->array([[1,2,3],[4,5,6]]);
        [$m,$n,$k,$AA,$offA,$BB,$offB] =
            $this->translate_reduceSum($A,$axis=1,$X);

        $BB = new \stdClass();
        $this->expectException(TypeError::class);
        $this->expectExceptionMessage('must be of type Interop\Polite\Math\Matrix\LinearBuffer');
        $matlib->reduceSum($m,$n,$k,$AA,$offA,$BB,$offB);
    }

    public function testreduceSumOverflowBufferBwithSize()
    {
        if($this->checkSkip('reduceSum')){return;}

        $matlib = $this->getMatlib();

        $X = $this->array([0,0]);
        $A = $this->array([[1,2,3],[4,5,6]]);
        [$m,$n,$k,$AA,$offA,$BB,$offB] =
            $this->translate_reduceSum($A,$axis=1,$X);

        $BB = $this->array([1])->buffer();
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Matrix specification too large for bufferB');
        $matlib->reduceSum($m,$n,$k,$AA,$offA,$BB,$offB);
    }

    public function testreduceSumOverflowBufferBwithOffsetB()
    {
        if($this->checkSkip('reduceSum')){return;}

        $matlib = $this->getMatlib();

        $X = $this->array([0,0]);
        $A = $this->array([[1,2,3],[4,5,6]]);
        [$m,$n,$k,$AA,$offA,$BB,$offB] =
            $this->translate_reduceSum($A,$axis=1,$X);

        $offB = 1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Matrix specification too large for bufferB');
        $matlib->reduceSum($m,$n,$k,$AA,$offA,$BB,$offB);
    }

    public function testreduceSumMinusOffsetA()
    {
        if($this->checkSkip('reduceSum')){return;}

        $matlib = $this->getMatlib();

        $X = $this->array([0,0]);
        $A = $this->array([[1,2,3],[4,5,6]]);
        [$m,$n,$k,$AA,$offA,$BB,$offB] =
            $this->translate_reduceSum($A,$axis=1,$X);

        $offA = -1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Argument offsetA must be greater than or equals 0.');
        $matlib->reduceSum($m,$n,$k,$AA,$offA,$BB,$offB);
    }

    public function testreduceSumIllegalBufferA()
    {
        if($this->checkSkip('reduceSum')){return;}

        $matlib = $this->getMatlib();

        $X = $this->array([0,0]);
        $A = $this->array([[1,2,3],[4,5,6]]);
        [$m,$n,$k,$AA,$offA,$BB,$offB] =
            $this->translate_reduceSum($A,$axis=1,$X);

        $AA = new \stdClass();
        $this->expectException(TypeError::class);
        $this->expectExceptionMessage('must be of type Interop\Polite\Math\Matrix\LinearBuffer');
        $matlib->reduceSum($m,$n,$k,$AA,$offA,$BB,$offB);
    }

    public function testreduceSumOverflowBufferAwithSize()
    {
        if($this->checkSkip('reduceSum')){return;}

        $matlib = $this->getMatlib();

        $X = $this->array([0,0]);
        $A = $this->array([[1,2,3],[4,5,6]]);
        [$m,$n,$k,$AA,$offA,$BB,$offB] =
            $this->translate_reduceSum($A,$axis=1,$X);

        $AA = $this->array([1,2,3,4,5])->buffer();
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Matrix specification too large for bufferA');
        $matlib->reduceSum($m,$n,$k,$AA,$offA,$BB,$offB);
    }

    public function testreduceSumOverflowBufferXwithOffsetA()
    {
        if($this->checkSkip('reduceSum')){return;}

        $matlib = $this->getMatlib();

        $X = $this->array([0,0]);
        $A = $this->array([[1,2,3],[4,5,6]]);
        [$m,$n,$k,$AA,$offA,$BB,$offB] =
            $this->translate_reduceSum($A,$axis=1,$X);

        $offA = 1;
        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('Matrix specification too large for bufferA');
        $matlib->reduceSum($m,$n,$k,$AA,$offA,$BB,$offB);
    }

    public function testsoftmax()
    {
        $matlib = $this->getMatlib();
        $X = $this->array([-1.0,-0.5,0.0,0.5,1.0]);
        $m = 1;
        $n = 5;
        $XX = $X->buffer();
        $offX = 0;
        $ldX = 5;
        $matlib->softmax($m,$n,$XX,$offX,$ldX);

        $this->assertTrue($X[0]>0.0);
        $this->assertTrue($X[0]<$X[1]);
        $this->assertTrue($X[1]<$X[2]);
        $this->assertTrue($X[2]<$X[3]);
        $this->assertTrue($X[3]<$X[4]);
        $this->assertTrue($X[4]<1.0);
        $this->assertTrue(1.0e-5>abs(1.0-$this->sum($n,$X,$offX,1)));
        $single = $X->toArray();

        // batch mode
        $y = $this->array([
            [-1.0,-0.5,0.0,0.5,1.0],
            [-1.0,-0.5,0.0,0.5,1.0],
            [-1.0,-0.5,0.0,0.5,1.0],
            [-1.0,-0.5,0.0,0.5,1.0],
            [-1.0,-0.5,0.0,0.5,1.0],
        ]);

        $m = 5;
        $n = 5;
        $YY = $y->buffer();
        $offY = 0;
        $ldY = 5;
        $matlib->softmax($m,$n,$YY,$offY,$ldY);

        $this->assertEquals($single,$y[0]->toArray());
        $this->assertEquals($single,$y[1]->toArray());
        $this->assertEquals($single,$y[2]->toArray());
        $this->assertEquals($single,$y[3]->toArray());
        $this->assertEquals($single,$y[4]->toArray());
    }

    public function testequal()
    {
        $matlib = $this->getMatlib();
        $n = 5;
        $offX = 0;
        $incX = 1;
        $offY = 0;
        $incY = 1;

        $X = $this->array([-1.0,-0.5,0.0,0.5,-1.0],NDArray::float32);
        $Y = $this->array([1.0,-0.5,0.0,0.5,1.0],NDArray::float32);
        $XX = $X->buffer();
        $YY = $Y->buffer();
        $matlib->equal($n,$XX,$offX,$incX,$YY,$offY,$incY);
        $this->assertEquals([0,1,1,1,0],$Y->toArray());

        $X = $this->array([-1.0,-0.5,0.0,0.5,-1.0],NDArray::float64);
        $Y = $this->array([1.0,-0.5,0.0,0.5,1.0],NDArray::float64);
        $XX = $X->buffer();
        $YY = $Y->buffer();
        $matlib->equal($n,$XX,$offX,$incX,$YY,$offY,$incY);
        $this->assertEquals([0,1,1,1,0],$Y->toArray());

        $X = $this->array([-2,-1,0,1,-1],NDArray::int8);
        $Y = $this->array([ 1,-1,0,1, 1],NDArray::int8);
        $XX = $X->buffer();
        $YY = $Y->buffer();
        $matlib->equal($n,$XX,$offX,$incX,$YY,$offY,$incY);
        $this->assertEquals([0,1,1,1,0],$Y->toArray());

        $X = $this->array([-2,-1,0,1,-1],NDArray::int16);
        $Y = $this->array([ 1,-1,0,1, 1],NDArray::int16);
        $XX = $X->buffer();
        $YY = $Y->buffer();
        $matlib->equal($n,$XX,$offX,$incX,$YY,$offY,$incY);
        $this->assertEquals([0,1,1,1,0],$Y->toArray());

        $X = $this->array([-2,-1,0,1,-1],NDArray::int32);
        $Y = $this->array([ 1,-1,0,1, 1],NDArray::int32);
        $XX = $X->buffer();
        $YY = $Y->buffer();
        $matlib->equal($n,$XX,$offX,$incX,$YY,$offY,$incY);
        $this->assertEquals([0,1,1,1,0],$Y->toArray());

        $X = $this->array([-2,-1,0,1,-1],NDArray::int64);
        $Y = $this->array([ 1,-1,0,1, 1],NDArray::int64);
        $XX = $X->buffer();
        $YY = $Y->buffer();
        $matlib->equal($n,$XX,$offX,$incX,$YY,$offY,$incY);
        $this->assertEquals([0,1,1,1,0],$Y->toArray());

        $X = $this->array([false,false,true ,true,true ],NDArray::bool);
        $Y = $this->array([true ,false,true ,true,false],NDArray::bool);
        $XX = $X->buffer();
        $YY = $Y->buffer();
        $matlib->equal($n,$XX,$offX,$incX,$YY,$offY,$incY);
        $this->assertEquals([false,true,true,true,false],$Y->toArray());
    }

    public function testnotequalNormal()
    {
        $matlib = $this->getMatlib();
        $n = 5;
        $offX = 0;
        $incX = 1;
        $offY = 0;
        $incY = 1;

        $X = $this->array([-1.0,-0.5,0.0,0.5,-1.0],NDArray::float32);
        $Y = $this->array([1.0,-0.5,0.0,0.5,1.0],NDArray::float32);
        $XX = $X->buffer();
        $YY = $Y->buffer();
        $matlib->notequal($n,$XX,$offX,$incX,$YY,$offY,$incY);
        $this->assertEquals([1,0,0,0,1],$Y->toArray());

        $X = $this->array([-1.0,-0.5,0.0,0.5,-1.0],NDArray::float64);
        $Y = $this->array([1.0,-0.5,0.0,0.5,1.0],NDArray::float64);
        $XX = $X->buffer();
        $YY = $Y->buffer();
        $matlib->notequal($n,$XX,$offX,$incX,$YY,$offY,$incY);
        $this->assertEquals([1,0,0,0,1],$Y->toArray());

        $X = $this->array([-2,-1,0,1,-1],NDArray::int8);
        $Y = $this->array([ 1,-1,0,1, 1],NDArray::int8);
        $XX = $X->buffer();
        $YY = $Y->buffer();
        $matlib->notequal($n,$XX,$offX,$incX,$YY,$offY,$incY);
        $this->assertEquals([1,0,0,0,1],$Y->toArray());

        $X = $this->array([-2,-1,0,1,-1],NDArray::int16);
        $Y = $this->array([ 1,-1,0,1, 1],NDArray::int16);
        $XX = $X->buffer();
        $YY = $Y->buffer();
        $matlib->notequal($n,$XX,$offX,$incX,$YY,$offY,$incY);
        $this->assertEquals([1,0,0,0,1],$Y->toArray());

        $X = $this->array([-2,-1,0,1,-1],NDArray::int32);
        $Y = $this->array([ 1,-1,0,1, 1],NDArray::int32);
        $XX = $X->buffer();
        $YY = $Y->buffer();
        $matlib->notequal($n,$XX,$offX,$incX,$YY,$offY,$incY);
        $this->assertEquals([1,0,0,0,1],$Y->toArray());

        $X = $this->array([-2,-1,0,1,-1],NDArray::int64);
        $Y = $this->array([ 1,-1,0,1, 1],NDArray::int64);
        $XX = $X->buffer();
        $YY = $Y->buffer();
        $matlib->notequal($n,$XX,$offX,$incX,$YY,$offY,$incY);
        $this->assertEquals([1,0,0,0,1],$Y->toArray());

        $X = $this->array([false,false,true ,true,true ],NDArray::bool);
        $Y = $this->array([true ,false,true ,true,false],NDArray::bool);
        $XX = $X->buffer();
        $YY = $Y->buffer();
        $matlib->notequal($n,$XX,$offX,$incX,$YY,$offY,$incY);
        $this->assertEquals([true,false,false,false,true],$Y->toArray());
    }

    public function testnotNormal()
    {
        $matlib = $this->getMatlib();
        $n = 5;
        $offX = 0;
        $incX = 1;

        $X = $this->array([-1.0,-0.5,0.0,0.5,-1.0],NDArray::float32);
        $XX = $X->buffer();
        $matlib->not($n,$XX,$offX,$incX);
        $this->assertEquals([0,0,1,0,0],$X->toArray());

        $X = $this->array([-1.0,-0.5,0.0,0.5,-1.0],NDArray::float64);
        $XX = $X->buffer();
        $matlib->not($n,$XX,$offX,$incX);
        $this->assertEquals([0,0,1,0,0],$X->toArray());

        $X = $this->array([-2,-1,0,1,-1],NDArray::int8);
        $XX = $X->buffer();
        $matlib->not($n,$XX,$offX,$incX);
        $this->assertEquals([0,0,1,0,0],$X->toArray());

        $X = $this->array([-2,-1,0,1,-1],NDArray::int16);
        $XX = $X->buffer();
        $matlib->not($n,$XX,$offX,$incX);
        $this->assertEquals([0,0,1,0,0],$X->toArray());

        $X = $this->array([-2,-1,0,1,-1],NDArray::int32);
        $XX = $X->buffer();
        $matlib->not($n,$XX,$offX,$incX);
        $this->assertEquals([0,0,1,0,0],$X->toArray());

        $X = $this->array([-2,-1,0,1,-1],NDArray::int64);
        $XX = $X->buffer();
        $matlib->not($n,$XX,$offX,$incX);
        $this->assertEquals([0,0,1,0,0],$X->toArray());

        $X = $this->array([false,false,true ,true,true ],NDArray::bool);
        $XX = $X->buffer();
        $matlib->not($n,$XX,$offX,$incX);
        $this->assertEquals([true,true,false,false,false],$X->toArray());
    }

    public function testastypeNormal()
    {
        $matlib = $this->getMatlib();

        #### int to any
        $X = $this->array([-1,0,1,2,3],NDArray::int32);
        $dtype = NDArray::float32;
        $Y = $this->zeros($X->shape(),$dtype);
        [$n,$dtype,$XX,$offX,$incX,$YY,$offY,$incY] = $this->translate_astype($X, $dtype, $Y);
        $matlib->astype($n,$dtype,$XX,$offX,$incX,$YY,$offY,$incY);
        $this->assertEquals([-1,0,1,2,3],$Y->toArray());

        $dtype = NDArray::float64;
        $Y = $this->zeros($X->shape(),$dtype);
        [$n,$dtype,$XX,$offX,$incX,$YY,$offY,$incY] = $this->translate_astype($X, $dtype, $Y);
        $matlib->astype($n,$dtype,$XX,$offX,$incX,$YY,$offY,$incY);
        $this->assertEquals([-1,0,1,2,3],$Y->toArray());

        $dtype = NDArray::int8;
        $Y = $this->zeros($X->shape(),$dtype);
        [$n,$dtype,$XX,$offX,$incX,$YY,$offY,$incY] = $this->translate_astype($X, $dtype, $Y);
        $matlib->astype($n,$dtype,$XX,$offX,$incX,$YY,$offY,$incY);
        $this->assertEquals([-1,0,1,2,3],$Y->toArray());

        $dtype = NDArray::int16;
        $Y = $this->zeros($X->shape(),$dtype);
        [$n,$dtype,$XX,$offX,$incX,$YY,$offY,$incY] = $this->translate_astype($X, $dtype, $Y);
        $matlib->astype($n,$dtype,$XX,$offX,$incX,$YY,$offY,$incY);
        $this->assertEquals([-1,0,1,2,3],$Y->toArray());

        $dtype = NDArray::int32;
        $Y = $this->zeros($X->shape(),$dtype);
        [$n,$dtype,$XX,$offX,$incX,$YY,$offY,$incY] = $this->translate_astype($X, $dtype, $Y);
        $matlib->astype($n,$dtype,$XX,$offX,$incX,$YY,$offY,$incY);
        $this->assertEquals([-1,0,1,2,3],$Y->toArray());

        $dtype = NDArray::int64;
        $Y = $this->zeros($X->shape(),$dtype);
        [$n,$dtype,$XX,$offX,$incX,$YY,$offY,$incY] = $this->translate_astype($X, $dtype, $Y);
        $matlib->astype($n,$dtype,$XX,$offX,$incX,$YY,$offY,$incY);
        $this->assertEquals([-1,0,1,2,3],$Y->toArray());

        $dtype = NDArray::bool;
        $Y = $this->zeros($X->shape(),$dtype);
        [$n,$dtype,$XX,$offX,$incX,$YY,$offY,$incY] = $this->translate_astype($X, $dtype, $Y);
        $matlib->astype($n,$dtype,$XX,$offX,$incX,$YY,$offY,$incY);
        $this->assertEquals([true,false,true,true,true],$Y->toArray());

        #### float to any ######
        $X = $this->array([-1,0,1,2,3],NDArray::float32);
        $dtype = NDArray::float32;
        $Y = $this->zeros($X->shape(),$dtype);
        [$n,$dtype,$XX,$offX,$incX,$YY,$offY,$incY] = $this->translate_astype($X, $dtype, $Y);
        $matlib->astype($n,$dtype,$XX,$offX,$incX,$YY,$offY,$incY);
        $this->assertEquals([-1,0,1,2,3],$Y->toArray());

        $dtype = NDArray::float64;
        $Y = $this->zeros($X->shape(),$dtype);
        [$n,$dtype,$XX,$offX,$incX,$YY,$offY,$incY] = $this->translate_astype($X, $dtype, $Y);
        $matlib->astype($n,$dtype,$XX,$offX,$incX,$YY,$offY,$incY);
        $this->assertEquals([-1,0,1,2,3],$Y->toArray());

        $dtype = NDArray::int8;
        $Y = $this->zeros($X->shape(),$dtype);
        [$n,$dtype,$XX,$offX,$incX,$YY,$offY,$incY] = $this->translate_astype($X, $dtype, $Y);
        $matlib->astype($n,$dtype,$XX,$offX,$incX,$YY,$offY,$incY);
        $this->assertEquals([-1,0,1,2,3],$Y->toArray());

        $dtype = NDArray::int16;
        $Y = $this->zeros($X->shape(),$dtype);
        [$n,$dtype,$XX,$offX,$incX,$YY,$offY,$incY] = $this->translate_astype($X, $dtype, $Y);
        $matlib->astype($n,$dtype,$XX,$offX,$incX,$YY,$offY,$incY);
        $this->assertEquals([-1,0,1,2,3],$Y->toArray());

        $dtype = NDArray::int32;
        $Y = $this->zeros($X->shape(),$dtype);
        [$n,$dtype,$XX,$offX,$incX,$YY,$offY,$incY] = $this->translate_astype($X, $dtype, $Y);
        $matlib->astype($n,$dtype,$XX,$offX,$incX,$YY,$offY,$incY);
        $this->assertEquals([-1,0,1,2,3],$Y->toArray());

        $dtype = NDArray::int64;
        $Y = $this->zeros($X->shape(),$dtype);
        [$n,$dtype,$XX,$offX,$incX,$YY,$offY,$incY] = $this->translate_astype($X, $dtype, $Y);
        $matlib->astype($n,$dtype,$XX,$offX,$incX,$YY,$offY,$incY);
        $this->assertEquals([-1,0,1,2,3],$Y->toArray());

        $dtype = NDArray::bool;
        $Y = $this->zeros($X->shape(),$dtype);
        [$n,$dtype,$XX,$offX,$incX,$YY,$offY,$incY] = $this->translate_astype($X, $dtype, $Y);
        $matlib->astype($n,$dtype,$XX,$offX,$incX,$YY,$offY,$incY);
        $this->assertEquals([true,false,true,true,true],$Y->toArray());

        #### bool to any ######
        $X = $this->array([true,false,true,true,true],NDArray::bool);
        $dtype = NDArray::float32;
        $Y = $this->zeros($X->shape(),$dtype);
        [$n,$dtype,$XX,$offX,$incX,$YY,$offY,$incY] = $this->translate_astype($X, $dtype, $Y);
        $matlib->astype($n,$dtype,$XX,$offX,$incX,$YY,$offY,$incY);
        $this->assertEquals([1,0,1,1,1],$Y->toArray());

        $dtype = NDArray::float64;
        $Y = $this->zeros($X->shape(),$dtype);
        [$n,$dtype,$XX,$offX,$incX,$YY,$offY,$incY] = $this->translate_astype($X, $dtype, $Y);
        $matlib->astype($n,$dtype,$XX,$offX,$incX,$YY,$offY,$incY);
        $this->assertEquals([1,0,1,1,1],$Y->toArray());

        $dtype = NDArray::int8;
        $Y = $this->zeros($X->shape(),$dtype);
        [$n,$dtype,$XX,$offX,$incX,$YY,$offY,$incY] = $this->translate_astype($X, $dtype, $Y);
        $matlib->astype($n,$dtype,$XX,$offX,$incX,$YY,$offY,$incY);
        $this->assertEquals([1,0,1,1,1],$Y->toArray());

        $dtype = NDArray::int16;
        $Y = $this->zeros($X->shape(),$dtype);
        [$n,$dtype,$XX,$offX,$incX,$YY,$offY,$incY] = $this->translate_astype($X, $dtype, $Y);
        $matlib->astype($n,$dtype,$XX,$offX,$incX,$YY,$offY,$incY);
        $this->assertEquals([1,0,1,1,1],$Y->toArray());

        $dtype = NDArray::int32;
        $Y = $this->zeros($X->shape(),$dtype);
        [$n,$dtype,$XX,$offX,$incX,$YY,$offY,$incY] = $this->translate_astype($X, $dtype, $Y);
        $matlib->astype($n,$dtype,$XX,$offX,$incX,$YY,$offY,$incY);
        $this->assertEquals([1,0,1,1,1],$Y->toArray());

        $dtype = NDArray::int64;
        $Y = $this->zeros($X->shape(),$dtype);
        [$n,$dtype,$XX,$offX,$incX,$YY,$offY,$incY] = $this->translate_astype($X, $dtype, $Y);
        $matlib->astype($n,$dtype,$XX,$offX,$incX,$YY,$offY,$incY);
        $this->assertEquals([1,0,1,1,1],$Y->toArray());

        $dtype = NDArray::bool;
        $Y = $this->zeros($X->shape(),$dtype);
        [$n,$dtype,$XX,$offX,$incX,$YY,$offY,$incY] = $this->translate_astype($X, $dtype, $Y);
        $matlib->astype($n,$dtype,$XX,$offX,$incX,$YY,$offY,$incY);
        $this->assertEquals([true,false,true,true,true],$Y->toArray());

        #### float to unsigned ######
        $X = $this->array([-1,0,1,2,3],NDArray::float32);
        $dtype = NDArray::uint8;
        $Y = $this->zeros($X->shape(),$dtype);
        [$n,$dtype,$XX,$offX,$incX,$YY,$offY,$incY] = $this->translate_astype($X, $dtype, $Y);
        $matlib->astype($n,$dtype,$XX,$offX,$incX,$YY,$offY,$incY);
        $this->assertEquals([255,0,1,2,3],$Y->toArray());

        $dtype = NDArray::uint16;
        $Y = $this->zeros($X->shape(),$dtype);
        [$n,$dtype,$XX,$offX,$incX,$YY,$offY,$incY] = $this->translate_astype($X, $dtype, $Y);
        $matlib->astype($n,$dtype,$XX,$offX,$incX,$YY,$offY,$incY);
        $this->assertEquals([65535,0,1,2,3],$Y->toArray());

        $dtype = NDArray::uint32;
        $Y = $this->zeros($X->shape(),$dtype);
        [$n,$dtype,$XX,$offX,$incX,$YY,$offY,$incY] = $this->translate_astype($X, $dtype, $Y);
        $matlib->astype($n,$dtype,$XX,$offX,$incX,$YY,$offY,$incY);
        $this->assertEquals([4294967295,0,1,2,3],$Y->toArray());

        $dtype = NDArray::uint64;
        $Y = $this->zeros($X->shape(),$dtype);
        [$n,$dtype,$XX,$offX,$incX,$YY,$offY,$incY] = $this->translate_astype($X, $dtype, $Y);
        $matlib->astype($n,$dtype,$XX,$offX,$incX,$YY,$offY,$incY);
        $this->assertEquals([-1,0,1,2,3],$Y->toArray());
    }

    #[DataProvider('providerDtypesFloats')]
    public function testreduceMaxSameSizeNormal($params)
    {
        extract($params);
        if($this->checkSkip('reduceMax')){return;}

        $matlib = $this->getMatlib();

        $A = $this->array([[1,2,3],[4,5,6]],dtype:$dtype);
        $X = $this->array([0,0],dtype:$dtype);
        [$m,$n,$k,$AA,$offA,$BB,$offB] =
            $this->translate_reduceSum($A,$axis=1,$X);

        $matlib->reduceMax($m,$n,$k,$AA,$offA,$BB,$offB);
        $this->assertEquals([3,6],$X->toArray());
    }

    #[DataProvider('providerDtypesFloats')]
    public function testmatrixcopyNormal($params)
    {
        extract($params);
        $matlib = $this->getMatlib();

        $A = $this->array([
            [1,2,3],
            [4,5,6],
        ],dtype:$dtype);
        $B = $this->zeros([3,2],dtype:$dtype);
        [$trans,$M,$N,$alpha,$AA,$offA,$ldA,$BB,$offB,$ldB] =
            $this->translate_matrixcopy($A,true,-1,$B);

        $this->assertEquals(2,$M);
        $this->assertEquals(3,$N);
        $matlib->matrixcopy(
            $trans,$M,$N,$alpha,$AA, $offA, $ldA,$BB, $offB, $ldB);
        $this->assertEquals([
            [-1,-4],
            [-2,-5],
            [-3,-6],
        ],$B->toArray());
    }

    #[DataProvider('providerDtypesFloats')]
    public function testImagecopyNormal($params)
    {
        extract($params);
        $matlib = $this->getMatlib();

        $a = $this->array([
            [[0],[1],[2]],
            [[3],[4],[5]],
            [[6],[7],[8]],
        ],dtype:$dtype);
        $b = $this->zeros($a->shape(),$a->dtype());
        [
            $height,
            $width,
            $channels,
            $AA, $offA,
            $BB, $offB,
            $channels_first,
            $heightShift,
            $widthShift,
            $verticalFlip,
            $horizontalFlip,
            $rgbFlip
        ] = $this->translate_imagecopy($a,B:$b,heightShift:1);

        $matlib->imagecopy(
            $height,
            $width,
            $channels,
            $AA, $offA,
            $BB, $offB,
            $channels_first,
            $heightShift,
            $widthShift,
            $verticalFlip,
            $horizontalFlip,
            $rgbFlip
        );
        $this->assertEquals([
            [[0],[1],[2]],
            [[0],[1],[2]],
            [[3],[4],[5]],
        ],$b->toArray());
    }

    #[DataProvider('providerDtypesFloats')]
    public function testreduceArgMaxSameSizeNormal($params)
    {
        extract($params);
        if($this->checkSkip('reduceArgMax')){return;}

        $matlib = $this->getMatlib();

        $A = $this->array([[1,2,3],[4,5,6]],dtype:$dtype);
        $X = $this->array([0,0],NDArray::int32);
        [$m,$n,$k,$AA,$offA,$BB,$offB] =
            $this->translate_reduceSum($A,$axis=1,$X);

        $matlib->reduceArgMax($m,$n,$k,$AA,$offA,$BB,$offB);
        $this->assertEquals([2,2],$X->toArray());
    }

    public function testRandomUniform()
    {
        $matlib = $this->getMatlib();

        // float32
        $x = $this->zeros([20,30]);
        [
            $n,
            $XX,$offX,$incX,
            $low,
            $high,
            $seed
        ] = $this->translate_randomUniform(
            $shape=[20,30],
            $low=-1.0,
            $high=1.0,
            output:$x,
        );
        $matlib->randomUniform(
            $n,
            $XX,$offX,$incX,
            $low,
            $high,
            $seed
        );

        $y = $this->zeros([20,30]);
        [
            $n,
            $XX,$offX,$incX,
            $low,
            $high,
            $seed
        ] = $this->translate_randomUniform(
            $shape=[20,30],
            $low=-1,
            $high=1,
            output:$y
        );
        $matlib->randomUniform(
            $n,
            $XX,$offX,$incX,
            $low,
            $high,
            $seed
        );

        $this->assertEquals(
            NDArray::float32,$x->dtype());
        $this->assertNotEquals(
            $x->toArray(),
            $y->toArray());

        $max = $x->buffer()[$matlib->imax($x->count(),$x->buffer(),0,1)];
        $min = $x->buffer()[$matlib->imin($x->count(),$x->buffer(),0,1)];
        $this->assertLessThanOrEqual(1,$max);
        $this->assertGreaterThanOrEqual(-1,$min);

        // int32
        $x = $this->zeros([20,30],dtype:NDArray::int32);
        [
            $n,
            $XX,$offX,$incX,
            $low,
            $high,
            $seed
        ] = $this->translate_randomUniform(
            $shape=[20,30],
            $low=-1,
            $high=1,
            dtype:NDArray::int32,
            output:$x,
        );
        $matlib->randomUniform(
            $n,
            $XX,$offX,$incX,
            $low,
            $high,
            $seed
        );

        $y = $this->zeros([20,30],dtype:NDArray::int32);
        [
            $n,
            $XX,$offX,$incX,
            $low,
            $high,
            $seed
        ] = $this->translate_randomUniform(
            $shape=[20,30],
            $low=-1,
            $high=1,
            dtype:NDArray::int32,
            output:$y
        );
        $matlib->randomUniform(
            $n,
            $XX,$offX,$incX,
            $low,
            $high,
            $seed
        );
        
        $this->assertEquals(
            NDArray::int32,$x->dtype());
        $this->assertNotEquals(
            $x->toArray(),
            $y->toArray());
        $xx = $this->zeros([20,30],NDArray::float32);
        $matlib->astype($x->count(),NDArray::float32,$x->buffer(),0,1,$xx->buffer(),0,1);
        $max = $xx->buffer()[$matlib->imax($xx->count(),$xx->buffer(),0,1)];
        $min = $xx->buffer()[$matlib->imin($xx->count(),$xx->buffer(),0,1)];
        $this->assertEquals(1,$max);
        $this->assertEquals(-1,$min);
    }

    #[DataProvider('providerDtypesFloats')]
    public function testRandomNormal($params)
    {
        extract($params);
        $matlib = $this->getMatlib();

        $x = $this->zeros([20,30],dtype:$dtype);
        [
            $n,
            $XX,$offX,$incX,
            $mean,
            $scale,
            $seed
        ] = $this->translate_randomNormal(
            $shape=[20,30],
            $mean=0.0,
            $scale=1.0,
            output:$x
        );
        $matlib->randomNormal(
            $n,
            $XX,$offX,$incX,
            $mean,
            $scale,
            $seed
        );
        $y = $this->zeros([20,30],dtype:$dtype);
        [
            $n,
            $XX,$offX,$incX,
            $mean,
            $scale,
            $seed
        ] = $this->translate_randomNormal(
            $shape=[20,30],
            $mean=0.0,
            $scale=1.0,
            output:$y
        );
        $matlib->randomNormal(
            $n,
            $XX,$offX,$incX,
            $mean,
            $scale,
            $seed
        );
        $this->assertEquals($dtype,$x->dtype());
        $this->assertNotEquals(
            $x->toArray(),
            $y->toArray());
        $max = $x->buffer()[$matlib->imax($x->count(),$x->buffer(),0,1)];
        $min = $x->buffer()[$matlib->imin($x->count(),$x->buffer(),0,1)];
        $this->assertLessThanOrEqual(5,$max);
        $this->assertGreaterThanOrEqual(-5,$min);
    }

    public function testRandomSequence()
    {
        $matlib = $this->getMatlib();

        $x = $this->zeros([$base=500],NDArray::int32);
        [
            $n,
            $size,
            $XX,$offX,$incX,
            $seed
        ] = $this->translate_randomSequence(
            $base=500,
            $size=100,
            output:$x
        );
        $matlib->randomSequence(
            $n,
            $size,
            $XX,$offX,$incX,
            $seed
        );
        $x = $x[R(0,$size)];

        $y = $this->zeros([$base=500],NDArray::int32);
        [
            $n,
            $size,
            $XX,$offX,$incX,
            $seed
        ] = $this->translate_randomSequence(
            $base=500,
            $size=100,
            output:$y
        );
        $matlib->randomSequence(
            $n,
            $size,
            $XX,$offX,$incX,
            $seed
        );
        $y = $y[R(0,$size)];

        $this->assertEquals(
                NDArray::int32,$x->dtype());
        $this->assertEquals(
            [100],$x->shape());
        $this->assertNotEquals(
            $x->toArray(),
            $y->toArray());
    }

    #[DataProvider('providerDtypesFloats')]
    public function testIm2col1dNormal($params)
    {
        extract($params);
        if($this->checkSkip('im2col1d')){return;}

        $matlib = $this->getMatlib();

        $images = $this->array([1,2,3,4],dtype:$dtype);
        $cols = $this->zeros([1,2,3,1],dtype:$dtype);

        $images_offset = $images->offset();
        $images_size = $images->size();
        $images_buff = $images->buffer();
        $cols_buff = $cols->buffer();
        $cols_offset = $cols->offset();
        $cols_size = $cols->size();
        $matlib->im2col1d(
            $reverse=false,
            $images_buff,
            $images_offset,
            $images_size,
            $batches=1,
            $in_w=4,
            $channels=1,
            $filter_w=3,
            $stride_w=1,
            $padding=false,
            $channels_first=false,
            $dilation_w=1,
            $cols_channels_first=false,
            $cols_buff,
            $cols_offset,
            $cols_size
        );
        $this->assertEquals(
            [[[[1],[2],[3]],
              [[2],[3],[4]]]],
            $cols->toArray());
    }

    #[DataProvider('providerDtypesFloats')]
    public function testIm2col2dNormal($params)
    {
        extract($params);
        if($this->checkSkip('im2col2d')){return;}

        $matlib = $this->getMatlib();

        $reverse = false;
        $batches = 1;
        $im_h = 4;
        $im_w = 4;
        $channels = 3;
        $kernel_h = 3;
        $kernel_w = 3;
        $stride_h = 1;
        $stride_w = 1;
        $padding = false;
        $channels_first = false;
        $cols_channels_first=false;
        $cols = null;
        $out_h = 2;
        $out_w = 2;
        $images = $this->arange(
            $batches*
            $im_h*$im_w*
            $channels,
            null,null,
            dtype:$dtype
        )->reshape([
            $batches,
            $im_h,
            $im_w,
            $channels
        ]);
        $cols = $this->zeros(
            [
                $batches,
                $out_h,$out_w,
                $kernel_h,$kernel_w,
                $channels,
            ],dtype:$dtype);
        $images_offset = $images->offset();
        $images_size = $images->size();
        $images_buff = $images->buffer();
        $cols_buff = $cols->buffer();
        $cols_offset = $cols->offset();
        $cols_size = $cols->size();
        $matlib->im2col2d(
            $reverse,
            $images_buff,
            $images_offset,
            $images_size,
            $batches,
            $im_h,
            $im_w,
            $channels,
            $kernel_h,
            $kernel_w,
            $stride_h,
            $stride_w,
            $padding,
            $channels_first,
            $dilation_h=1,
            $dilation_w=1,
            $cols_channels_first,
            $cols_buff,
            $cols_offset,
            $cols_size
        );
        $this->assertEquals(
        [[
          [
           [[[0,1,2],[3,4,5],[6,7,8]],
            [[12,13,14],[15,16,17],[18,19,20]],
            [[24,25,26],[27,28,29],[30,31,32]],],
           [[[3,4,5],[6,7,8],[9,10,11]],
            [[15,16,17],[18,19,20],[21,22,23]],
            [[27,28,29],[30,31,32],[33,34,35]],],
          ],
          [
           [[[12,13,14],[15,16,17],[18,19,20]],
            [[24,25,26],[27,28,29],[30,31,32]],
            [[36,37,38],[39,40,41],[42,43,44]],],
           [[[15,16,17],[18,19,20],[21,22,23]],
            [[27,28,29],[30,31,32],[33,34,35]],
            [[39,40,41],[42,43,44],[45,46,47]],],
          ],
        ]],
        $cols->toArray()
        );
    }

    #[DataProvider('providerDtypesFloats')]
    public function testIm2col3dNormal($params)
    {
        extract($params);
        if($this->checkSkip('im2col3d')){return;}

        $matlib = $this->getMatlib();

        $reverse = false;
        $batches = 1;
        $im_d = 4;
        $im_h = 4;
        $im_w = 4;
        $channels = 3;
        $kernel_d = 3;
        $kernel_h = 3;
        $kernel_w = 3;
        $stride_d = 1;
        $stride_h = 1;
        $stride_w = 1;
        $padding = false;
        $channels_first = false;
        $cols_channels_first=false;
        $cols = null;
        $out_d = 2;
        $out_h = 2;
        $out_w = 2;

        $images = $this->arange(
            $batches*
            $im_d*$im_h*$im_w*
            $channels,
            null,null,
            dtype:$dtype
        )->reshape([
            $batches,
            $im_d,
            $im_h,
            $im_w,
            $channels
        ]);

        $cols = $this->zeros(
            [
                $batches,
                $out_d,$out_h,$out_w,
                $kernel_d,$kernel_h,$kernel_w,
                $channels,
            ],dtype:$dtype);
        $images_offset = $images->offset();
        $images_size = $images->size();
        $images_buff = $images->buffer();
        $cols_buff = $cols->buffer();
        $cols_offset = $cols->offset();
        $cols_size = $cols->size();
        $matlib->im2col3d(
            $reverse,
            $images_buff,
            $images_offset,
            $images_size,
            $batches,
            $im_d,
            $im_h,
            $im_w,
            $channels,
            $kernel_d,
            $kernel_h,
            $kernel_w,
            $stride_d,
            $stride_h,
            $stride_w,
            $padding,
            $channels_first,
            $dilation_d=1,
            $dilation_h=1,
            $dilation_w=1,
            $cols_channels_first,
            $cols_buff,
            $cols_offset,
            $cols_size
        );
        $this->assertNotEquals(
            $cols->toArray(),
            $this->zerosLike($cols)
            );
    }

    ///**
    //* @dataProvider providerDtypesFloats
    //*/
    //public function testMaskingSameSizeNormal($params)
    //{
    //    extract($params);
    //    if($this->checkSkip('masking')){return;}
//
    //    $matlib = $this->getMatlib();
//
    //    $X = $this->array([true,false,true],dtype:NDArray::bool);
    //    $A = $this->array([10,100,1000],dtype:$dtype);
    //    [$trans,$M,$N,$XX,$offX,$incX,$AA,$offA,$ldA] =
    //        $this->translate_multiply($X,$A);
//
    //    $matlib->masking($trans,$M,$N,$XX,$offX,$incX,$AA,$offA,$ldA);
    //    $this->assertEquals([10,0,1000],$A->toArray());
    //}
//
    ///**
    //* @dataProvider providerDtypesFloats
    //*/
    //public function testMaskingBroadcastNormal($params)
    //{
    //    extract($params);
    //    if($this->checkSkip('masking')){return;}
//
    //    $matlib = $this->getMatlib();
//
    //    $X = $this->array([true,false,true],dtype:NDArray::bool);
    //    $A = $this->array([[10,100,1000],[-1,-1,-1]],dtype:$dtype);
    //    [$trans,$M,$N,$XX,$offX,$incX,$AA,$offA,$ldA] =
    //        $this->translate_multiply($X,$A);
//
    //    $matlib->masking($trans,$M,$N,$XX,$offX,$incX,$AA,$offA,$ldA);
    //    $this->assertEquals([[10, 0,1000],[-1, 0,-1]],$A->toArray());
    //}
//
    //public function testMaskingBroadcastTranspose()
    //{
    //    if($this->checkSkip('masking')){return;}
//
    //    $matlib = $this->getMatlib();
//
    //    $X = $this->array([true,false,true],dtype:NDArray::bool);
    //    $A = $this->array([[10,100],[1000,10000],[-1,-1]]);
    //    [$trans,$M,$N,$XX,$offX,$incX,$AA,$offA,$ldA] =
    //        $this->translate_multiply($X,$A,true);
//
    //    $matlib->masking($trans,$M,$N,$XX,$offX,$incX,$AA,$offA,$ldA);
    //    $this->assertEquals([[10,100],[ 0, 0],[-1,-1]],$A->toArray());
    //}
//

    #[DataProvider('providerDtypesFloatsAndInteger3264')]
    public function testMaskingSameSizeNormal($params)
    {
        extract($params);
        if($this->checkSkip('masking')){return;}

        $matlib = $this->getMatlib();

        $X = $this->array([true,false,true],dtype:NDArray::bool);
        $A = $this->array([10,100,1000],dtype:$dtype);
        [$M,$N,$K,$LEN,$fill,$mode,$XX,$offX,$AA,$offA] =
            $this->translate_masking($X,$A);

        $matlib->masking($M,$N,$K,$LEN,$fill,$mode,$XX,$offX,$AA,$offA);
        $this->assertEquals([10,0,1000],$A->toArray());
    }

    #[DataProvider('providerDtypesFloatsAndInteger326w3246indexes')]
    public function testMaskingBroadcastWithGapAndLenNormal($params)
    {
        extract($params);
        if($this->checkSkip('masking')){return;}

        $matlib = $this->getMatlib();

        // broadcast with gap and implicit len
        // X:(2,  3  )
        // A:(2,4,3,2)
        // outer:(2),bro:(4),inner:(3),bro2:(2)
        // m=2,n=4,k=3,len=2
        $X = $this->array([
            [true,false,true],
            [false,true,false]
        ],dtype:NDArray::bool);
        $A = $this->array([
            [[[1,-1],[11,-11],[111,-111]],
             [[2,-2],[12,-12],[112,-112]],
             [[-3,3],[13,-13],[113,-113]],
             [[-4,4],[14,-14],[114,-114]]],
            [[[1,-1],[21,-21],[211,-211]],
             [[2,-2],[22,-22],[222,-222]],
             [[-3,3],[23,-23],[223,-223]],
             [[-4,4],[24,-24],[224,-224]]],
        ],dtype:$dtype);
        [$M,$N,$K,$LEN,$fill,$mode,$XX,$offX,$AA,$offA] =
            $this->translate_masking($X,$A,batchDims:1,axis:2);

        $matlib->masking($M,$N,$K,$LEN,$fill,$mode,$XX,$offX,$AA,$offA);
        $this->assertEquals([
            [[[1,-1],[ 0,  0],[111,-111]],
             [[2,-2],[ 0,  0],[112,-112]],
             [[-3,3],[ 0,  0],[113,-113]],
             [[-4,4],[ 0,  0],[114,-114]]],
            [[[0, 0],[21,-21],[  0,   0]],
             [[0, 0],[22,-22],[  0,   0]],
             [[0, 0],[23,-23],[  0,   0]],
             [[0, 0],[24,-24],[  0,   0]]],
        ],$A->toArray());
    }

    #[DataProvider('providerDtypesFloatsAndInteger326w3246indexes')]
    public function testMaskingMaskIsNotBoolType($params)
    {
        extract($params);
        if($this->checkSkip('masking')){return;}

        $matlib = $this->getMatlib();

        $X = $this->array([1,0,1],dtype:NDArray::uint8);
        $A = $this->array([10,100,1000],dtype:$dtype);
        [$M,$N,$K,$LEN,$fill,$mode,$XX,$offX,$AA,$offA] =
            $this->translate_masking($X,$A);

        $this->expectException(InvalidArgumentException::class);
        $this->expectExceptionMessage('dtype of X must be bool.: uint8');
        $matlib->masking($M,$N,$K,$LEN,$fill,$mode,$XX,$offX,$AA,$offA);
    }

    #[DataProvider('providerDtypesFloatsAndInteger326w3246indexes')]
    public function testMaskingAddMode($params)
    {
        extract($params);
        if($this->checkSkip('masking')){return;}

        $matlib = $this->getMatlib();

        $X = $this->array([true,false,true],dtype:NDArray::bool);
        $A = $this->array([10,100,1000],dtype:$dtype);
        [$M,$N,$K,$LEN,$fill,$mode,$XX,$offX,$AA,$offA] =
            $this->translate_masking($X,$A,fill:-10000,mode:1);

        $matlib->masking($M,$N,$K,$LEN,$fill,$mode,$XX,$offX,$AA,$offA);
        $this->assertEquals([10,-9900,1000],$A->toArray());
    }

    public function testMaskingBoolSet()
    {
        if($this->checkSkip('masking')){return;}

        $matlib = $this->getMatlib();
        $dtype = NDArray::bool;

        $X = $this->array([true, true, false,false,false,true ],dtype:NDArray::bool);
        $A = $this->array([true, false,true, false,true, true ],dtype:$dtype);
        [$M,$N,$K,$LEN,$fill,$mode,$XX,$offX,$AA,$offA] =
            $this->translate_masking($X,$A);

        $matlib->masking($M,$N,$K,$LEN,$fill,$mode,$XX,$offX,$AA,$offA);
        $this->assertEquals([true, false,false,false,false,true],$A->toArray());
    }

    public function testMaskingBoolAdd()
    {
        if($this->checkSkip('masking')){return;}

        $matlib = $this->getMatlib();
        $dtype = NDArray::bool;

        $X = $this->array([true, true, false,false,false,true ],dtype:NDArray::bool);
        $A = $this->array([true, false,true, false,true, true ],dtype:$dtype);
        [$M,$N,$K,$LEN,$fill,$mode,$XX,$offX,$AA,$offA] =
            $this->translate_masking($X,$A,fill:true,mode:1);

        $matlib->masking($M,$N,$K,$LEN,$fill,$mode,$XX,$offX,$AA,$offA);
        $this->assertEquals([true,false,true, true, true, true ],$A->toArray());
    }

    #[DataProvider('providerDtypesFloats')]
    public function testEinsumSimple($params)
    {
        extract($params);
        if($this->checkSkip('einsum')){return;}

        $matlib = $this->getMatlib();

        $equation = 'ab,bc->ac';
        $A = $this->array([
            [1,2,3],
            [4,5,6],
        ],dtype:$dtype);
        $B = $this->array([
            [1,2,3,4],
            [5,6,7,8],
            [9,10,11,12],
        ],dtype:$dtype);
        [
            $sizeOfIndices,
            $AA,
            $offsetA,
            $ldA,
            $BB,
            $offsetB,
            $ldB,
            $CC,
            $offsetC,
            $ndimC,
            $outputs,
        ] = $this->translate_einsum($equation,$A,$B);

        $matlib->einsum(
            $sizeOfIndices,
            $AA,
            $offsetA,
            $ldA,
            $BB,
            $offsetB,
            $ldB,
            $CC,
            $offsetC,
            $ndimC,
        );

        $this->assertEquals([
            [ 38,  44,  50,  56],
            [ 83,  98, 113, 128],
        ],$outputs->toArray());
    }

    #[DataProvider('providerDtypesFloats')]
    public function testEinsum4p1Simple($params)
    {
        extract($params);
        if($this->checkSkip('einsum4p1')){return;}

        $matlib = $this->getMatlib();

        $equation = 'ab,bc->ac';
        $A = $this->array([
            [1,2,3],
            [4,5,6],
        ],dtype:$dtype);
        $B = $this->array([
            [1,2,3,4],
            [5,6,7,8],
            [9,10,11,12],
        ],dtype:$dtype);
        [
            $dim0,$dim1,$dim2,$dim3,$dim4,
            $AA,
            $offsetA,
            $ldA0,$ldA1,$ldA2,$ldA3,$ldA4,
            $BB,
            $offsetB,
            $ldB0,$ldB1,$ldB2,$ldB3,$ldB4,
            $CC,
            $offsetC,
            $c,
        ] = $this->translate_einsum4p1($equation,$A,$B);

        $matlib->einsum4p1(
            $dim0,$dim1,$dim2,$dim3,$dim4,
            $AA,
            $offsetA,
            $ldA0,$ldA1,$ldA2,$ldA3,$ldA4,
            $BB,
            $offsetB,
            $ldB0,$ldB1,$ldB2,$ldB3,$ldB4,
            $CC,
            $offsetC,
        );

        $this->assertEquals([
            [ 38,  44,  50,  56],
            [ 83,  98, 113, 128],
        ],$c->toArray());
    }

}
