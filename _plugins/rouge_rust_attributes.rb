# Rouge's Rust lexer has no rule for generic brackets or lifetimes inside
# attributes such as `#[dispatch(walk, (AnyView<'_>, Kind), Result<T>)]`.
# It emits `<`, `>` and `'` there as error tokens, which the site styles in
# red on a pink background. Highlight them as part of the attribute instead.
require "rouge"

Rouge::Lexers::Rust.prepend :attribute do
  rule %r/[<>'&]/, Rouge::Token::Tokens::Name::Decorator
end
