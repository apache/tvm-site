/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

//! Complex values in the IR, lowered to ordinary scalar arithmetic.
//! See docs/advanced/extending_ir_in_rust.rst; the my_ext Python package loads this library.

use tvm_tirx::ir::op_attr::CALL_EFFECT_KIND;
use tvm_tirx::ir::prim::ex;
use tvm_tirx::ir::sig::{self, PrimExpr};
use tvm_tirx::ir::transform::{Pass, PassContext, PassInfo};
use tvm_tirx::ir::{
    dtype, register_type, Bind, Call, CallEffectKind, CallObj, Expr, IRModule, Op, OpDef, OpHandle,
    PrimType, SeqStmt, Stmt, Type, TypeObj, Var,
};
use tvm_tirx::tirx::transform::FunctionPass;
use tvm_tirx::tirx::Function;
use tvm_tirx::tvm_ffi::{
    self, def_object_ref, dispatch, rs_class, Error, ObjectRefCast, ObjectRefCore, Result,
    WalkOrder,
};

// begin type
/// A complex number whose real and imaginary parts share a scalar type.
#[rs_class("my_ext.ComplexType", final)]
pub struct ComplexTypeObj {
    base: TypeObj,
    pub elem: PrimType,
}
def_object_ref!(pub ComplexType, Type, ComplexTypeObj);

tvm_ffi::static_init_block! {
    register_type::<ComplexType>()?;
}
// end type

fn type_error(message: &str) -> Error {
    Error::new(tvm_ffi::TYPE_ERROR, message, "")
}

fn require(condition: bool, message: &str) -> Result<()> {
    if condition {
        Ok(())
    } else {
        Err(type_error(message))
    }
}

fn check_call(call: &CallObj, count: usize) -> Result<()> {
    require(call.args.len() == count, &format!("expected {count} value arguments"))?;
    require(call.ty_args.is_empty(), "type arguments are not supported")?;
    require(call.attrs.is_none(), "call attributes are not supported")
}

/// The type of a part of a complex number: a scalar float, such as `float32`, `float64` or
/// `bfloat16`.
fn float_part(ty: &Type) -> Result<PrimType> {
    let elem: PrimType =
        ty.clone().try_cast().map_err(|_| type_error("expected a scalar float"))?;
    let bits = elem.dtype.bits;
    let float = elem.dtype == dtype::float(bits) || elem.dtype == dtype::bfloat(bits);
    require(float, "expected a scalar float")?;
    Ok(elem)
}

// begin inference
fn infer_complex(call: &CallObj) -> Result<Type> {
    check_call(call, 2)?;
    let elem = float_part(&call.args.get(0)?.ty)?;
    let im = float_part(&call.args.get(1)?.ty)?;
    require(elem.dtype == im.dtype, "my_ext.complex expects parts of the same type")?;
    let ty: ComplexType = ComplexTypeObj::new(TypeObj::new(call.loc.clone()), elem).into();
    Ok(ty.into())
}
// end inference

fn complex_arg(call: &CallObj, index: usize) -> Result<ComplexType> {
    let op: Op = call.op.clone().try_cast()?;
    let ty: ComplexType = call
        .args
        .get(index)?
        .ty
        .clone()
        .try_cast()
        .map_err(|_| type_error(&format!("{} expects a complex value", op.name.as_str())))?;
    float_part(&ty.elem.clone().into())?;
    Ok(ty)
}

fn infer_cmul(call: &CallObj) -> Result<Type> {
    check_call(call, 2)?;
    let (a, b) = (complex_arg(call, 0)?, complex_arg(call, 1)?);
    require(a.elem.dtype == b.elem.dtype, "my_ext.cmul expects complex values of the same type")?;
    Ok(a.into())
}

fn infer_part(call: &CallObj) -> Result<Type> {
    check_call(call, 1)?;
    Ok(complex_arg(call, 0)?.elem.clone().into())
}

// Explicit result types bypass inference, so validation checks the inputs and result again.
fn validate_call(call: &CallObj) -> Result<()> {
    let expected = if COMPLEX.is(&call.op) {
        infer_complex(call)?
    } else if CMUL.is(&call.op) {
        infer_cmul(call)?
    } else {
        infer_part(call)?
    };
    let matches: bool = tvm_ffi::cached_global_func!("ffi.StructuralEqual")
        .call_tuple((&expected, &call.ty, false, false))?
        .try_into()?;
    require(matches, "incorrect result type for complex operator")
}

// begin constructor
tvm_ffi::static_init_block! {
    OpDef::new("my_ext.complex", "Construct the complex number re + im * i.")
        .signature((
            sig::arg::<PrimExpr>("re", "The real part."),
            sig::arg::<PrimExpr>("im", "The imaginary part."),
        ))
        .set_infer_type(infer_complex)
        .set_validator(validate_call)
        .set_attr(CALL_EFFECT_KIND, CallEffectKind::Pure);
}
// end constructor

tvm_ffi::static_init_block! {
    OpDef::new("my_ext.cmul", "Multiply two complex numbers.")
        .signature((
            sig::arg::<Expr>("a", "The first complex value."),
            sig::arg::<Expr>("b", "The second complex value."),
        ))
        .set_infer_type(infer_cmul)
        .set_validator(validate_call)
        .set_attr(CALL_EFFECT_KIND, CallEffectKind::Pure);
    OpDef::new("my_ext.real", "Extract the real part of a complex number.")
        .signature(sig::arg::<Expr>("z", "The complex value."))
        .set_infer_type(infer_part)
        .set_validator(validate_call)
        .set_attr(CALL_EFFECT_KIND, CallEffectKind::Pure);
    OpDef::new("my_ext.imag", "Extract the imaginary part of a complex number.")
        .signature(sig::arg::<Expr>("z", "The complex value."))
        .set_infer_type(infer_part)
        .set_validator(validate_call)
        .set_attr(CALL_EFFECT_KIND, CallEffectKind::Pure);
}

// begin handles
static COMPLEX: OpHandle = OpHandle::new("my_ext.complex");
static CMUL: OpHandle = OpHandle::new("my_ext.cmul");
static REAL: OpHandle = OpHandle::new("my_ext.real");
static IMAG: OpHandle = OpHandle::new("my_ext.imag");
// end handles

#[derive(Default)]
struct Lowering {
    // Each complex local variable is replaced by two scalar local variables.
    bindings: Vec<(Var, Var, Var)>,
}

impl Lowering {
    fn parts(&self, expr: &Expr) -> Result<(Expr, Expr)> {
        if let Some(call) = expr.as_node::<CallObj>() {
            if COMPLEX.is(&call.op) {
                return Ok((call.args.get(0)?, call.args.get(1)?));
            }
        }
        for (var, re, im) in &self.bindings {
            if var.same_as(expr) {
                return Ok((re.clone().into(), im.clone().into()));
            }
        }
        Err(Error::new(tvm_ffi::VALUE_ERROR, "LowerComplex cannot resolve complex operand", ""))
    }
}

#[dispatch(map)]
impl Lowering {
    fn map_sequence(&mut self, seq: SeqStmt) -> SeqStmt {
        SeqStmt::new(seq.seq.iter(), seq.loc.clone())
    }

    // begin lowering
    fn map_call(&mut self, call: Call) -> Result<Expr> {
        if COMPLEX.is(&call.op) || CMUL.is(&call.op) || REAL.is(&call.op) || IMAG.is(&call.op) {
            validate_call(&call)?;
        }
        if CMUL.is(&call.op) {
            let (a, b) = self.parts(&call.args.get(0)?)?;
            let (c, d) = self.parts(&call.args.get(1)?)?;
            // (a + bi)(c + di) = (ac - bd) + (ad + bc)i
            let re = ex(&a) * ex(&c) - ex(&b) * ex(&d);
            let im = ex(&a) * ex(&d) + ex(&b) * ex(&c);
            return Ok(Call::new(
                None,
                COMPLEX.get()?,
                [re, im],
                None,
                Vec::<Type>::new(),
                call.loc.clone(),
            )?
            .into());
        }
        if REAL.is(&call.op) || IMAG.is(&call.op) {
            let (re, im) = self.parts(&call.args.get(0)?)?;
            return Ok(if REAL.is(&call.op) { re } else { im });
        }
        Ok(call.into())
    }

    fn map_bind(&mut self, bind: Bind) -> Result<Stmt> {
        if let Some(ty) = bind.var.ty.as_node::<ComplexTypeObj>() {
            let (re, im) = self.parts(&bind.value)?;
            let name = bind.var.name.as_str();
            let re_var = Var::new(&format!("{name}_re"), ty.elem.clone(), bind.var.loc.clone());
            let im_var = Var::new(&format!("{name}_im"), ty.elem.clone(), bind.var.loc.clone());
            self.bindings.push((bind.var.clone(), re_var.clone(), im_var.clone()));
            let binds = [
                Bind::new(re_var, re, bind.loc.clone())?,
                Bind::new(im_var, im, bind.loc.clone())?,
            ];
            return Ok(SeqStmt::new(binds, bind.loc.clone()).into());
        }
        Ok(bind.into())
    }
    // end lowering
}

/// Reject a complex type, wherever it appears: on a variable, a parameter or an expression.
fn check_type(ty: Type) -> Result<Type> {
    if ty.as_node::<ComplexTypeObj>().is_some() {
        return Err(Error::new(
            tvm_ffi::VALUE_ERROR,
            "LowerComplex left an unsupported complex type",
            "",
        ));
    }
    Ok(ty)
}

/// Reject an expression whose type is complex.
fn check_expr(expr: Expr) -> Result<Expr> {
    check_type(expr.ty.clone())?;
    Ok(expr)
}

/// Reject a call to one of the four complex operators that the rewrite did not consume.
fn check_call_lowered(call: Call) -> Result<Call> {
    if COMPLEX.is(&call.op) || CMUL.is(&call.op) || REAL.is(&call.op) || IMAG.is(&call.op) {
        return Err(Error::new(
            tvm_ffi::VALUE_ERROR,
            "LowerComplex left an unsupported complex call",
            "",
        ));
    }
    Ok(call)
}

// begin function callback
fn lower_function(
    func: Function,
    _module: IRModule,
    _ctx: PassContext,
) -> Result<Option<Function>> {
    let mut lowering = Lowering::default();
    let rewritten = tvm_ffi::structural_map(func, &mut lowering, WalkOrder::PostOrder)?;
    let checked = tvm_ffi::structural_map(
        rewritten,
        (check_type, check_call_lowered, check_expr),
        WalkOrder::PostOrder,
    )?;
    let function: Function = checked.try_into()?;
    Ok(Some(function))
}
// end function callback

// begin pass
fn lower_complex() -> Result<Pass> {
    let pass = FunctionPass::new(lower_function, PassInfo::new(0, "my_ext.LowerComplex"));
    Ok(Pass::from(&pass))
}

tvm_ffi::tvm_ffi_dll_export_typed_func!(lower_complex, lower_complex);
// end pass
