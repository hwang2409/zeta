//! Markdown render tree, parsed once per text update without GPUI.
use crate::appearance::Appearance;
use pulldown_cmark::{CodeBlockKind, Event, Parser, Tag, TagEnd};
use std::{ops::Range, sync::LazyLock};
use syntect::{
    easy::HighlightLines, highlighting::ThemeSet, parsing::SyntaxSet, util::LinesWithEndings,
};

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Inline {
    pub text: String,
    pub bold: bool,
    pub italic: bool,
    pub code: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BlockKind {
    Document,
    Paragraph,
    Heading(u8),
    List(Option<u64>),
    Item,
    Quote,
    Code(String),
    Rule,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SyntaxRun {
    pub range: Range<usize>,
    pub light: u32,
    pub dark: u32,
}

impl SyntaxRun {
    pub fn color(&self, appearance: Appearance) -> u32 {
        appearance.syntax_color(match appearance {
            Appearance::Light => self.light,
            Appearance::Dark => self.dark,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Block {
    pub kind: BlockKind,
    pub spans: Vec<Inline>,
    pub children: Vec<Block>,
    // Foreground-only runs: theme backgrounds cannot enter the render tree.
    pub syntax: Vec<SyntaxRun>,
}

impl Block {
    fn new(kind: BlockKind) -> Self {
        Self {
            kind,
            spans: vec![],
            children: vec![],
            syntax: vec![],
        }
    }
    pub fn text(&self) -> String {
        self.spans.iter().map(|s| s.text.as_str()).collect()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Markdown {
    pub source: String,
    pub root: Block,
}

impl From<String> for Markdown {
    fn from(source: String) -> Self {
        let root = parse(&source);
        Self { source, root }
    }
}
impl From<&str> for Markdown {
    fn from(source: &str) -> Self {
        source.to_owned().into()
    }
}
impl Markdown {
    pub fn push_str(&mut self, delta: &str) {
        self.source.push_str(delta);
        self.root = parse(&self.source);
    }
}

fn parse(source: &str) -> Block {
    let mut stack = vec![Block::new(BlockKind::Document)];
    let mut bold = 0;
    let mut italic = 0;
    for event in Parser::new(source) {
        let kind = match &event {
            Event::Start(Tag::Paragraph) => Some(BlockKind::Paragraph),
            Event::Start(Tag::Heading { level, .. }) => Some(BlockKind::Heading(*level as u8)),
            Event::Start(Tag::List(start)) => Some(BlockKind::List(*start)),
            Event::Start(Tag::Item) => Some(BlockKind::Item),
            Event::Start(Tag::BlockQuote(_)) => Some(BlockKind::Quote),
            Event::Start(Tag::CodeBlock(kind)) => Some(BlockKind::Code(match kind {
                CodeBlockKind::Fenced(info) => info.split_whitespace().next().unwrap_or("").into(),
                CodeBlockKind::Indented => String::new(),
            })),
            _ => None,
        };
        if let Some(kind) = kind {
            stack.push(Block::new(kind));
            continue;
        }
        match event {
            Event::End(
                TagEnd::Paragraph
                | TagEnd::Heading(_)
                | TagEnd::List(_)
                | TagEnd::Item
                | TagEnd::BlockQuote(_)
                | TagEnd::CodeBlock,
            ) => {
                let mut block = stack.pop().expect("balanced markdown tags");
                if let BlockKind::Code(language) = &block.kind {
                    block.syntax = highlight(&block.text(), language);
                }
                stack.last_mut().unwrap().children.push(block);
            }
            Event::Start(Tag::Strong) => bold += 1,
            Event::End(TagEnd::Strong) => bold -= 1,
            Event::Start(Tag::Emphasis) => italic += 1,
            Event::End(TagEnd::Emphasis) => italic -= 1,
            Event::Text(text) | Event::Html(text) | Event::InlineHtml(text) => {
                stack.last_mut().unwrap().spans.push(Inline {
                    text: text.into_string(),
                    bold: bold > 0,
                    italic: italic > 0,
                    code: false,
                });
            }
            Event::Code(text) => stack.last_mut().unwrap().spans.push(Inline {
                text: text.into_string(),
                code: true,
                bold: bold > 0,
                italic: italic > 0,
            }),
            Event::SoftBreak | Event::HardBreak => stack.last_mut().unwrap().spans.push(Inline {
                text: "\n".into(),
                ..Default::default()
            }),
            Event::Rule => stack
                .last_mut()
                .unwrap()
                .children
                .push(Block::new(BlockKind::Rule)),
            _ => {}
        }
    }
    stack.pop().unwrap()
}

static SYNTAX: LazyLock<SyntaxSet> = LazyLock::new(SyntaxSet::load_defaults_newlines);
static THEMES: LazyLock<ThemeSet> = LazyLock::new(ThemeSet::load_defaults);

fn highlight(text: &str, language: &str) -> Vec<SyntaxRun> {
    let syntax = SYNTAX
        .find_syntax_by_token(language)
        .unwrap_or_else(|| SYNTAX.find_syntax_plain_text());
    let colors = |theme: &str| {
        let mut highlighter = HighlightLines::new(syntax, &THEMES.themes[theme]);
        let mut offset = 0;
        let mut runs = Vec::new();
        for line in LinesWithEndings::from(text) {
            let Ok(parts) = highlighter.highlight_line(line, &SYNTAX) else {
                // A syntax failure falls back to plain text for the whole fence.
                return Vec::new();
            };
            for (style, text) in parts {
                let color = style.foreground;
                runs.push((
                    offset..offset + text.len(),
                    (u32::from(color.r) << 16) | (u32::from(color.g) << 8) | u32::from(color.b),
                ));
                offset += text.len();
            }
        }
        runs
    };
    let light = colors("InspiredGitHub");
    let dark = colors("base16-ocean.dark");
    // Theme changes can coalesce different token boundaries. Merge their endpoints.
    let mut boundaries: Vec<_> = light
        .iter()
        .chain(&dark)
        .flat_map(|(r, _)| [r.start, r.end])
        .collect();
    boundaries.sort_unstable();
    boundaries.dedup();
    boundaries
        .windows(2)
        .map(|pair| SyntaxRun {
            range: pair[0]..pair[1],
            light: light
                .get(light.partition_point(|(range, _)| range.end <= pair[0]))
                .map_or(0x202024, |(_, c)| *c),
            dark: dark
                .get(dark.partition_point(|(range, _)| range.end <= pair[0]))
                .map_or(0xededf0, |(_, c)| *c),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::appearance::contrast;
    #[test]
    fn maps_headings_lists_inline_code_and_fences() {
        let doc = Markdown::from("# Title\n\n## Second\n\nplain **bold** `code`\n\n3. first\n4. second\n   - nested\n\n```rust\nfn main() { println!(\"hi\"); }\n```\n");
        let blocks = &doc.root.children;
        assert_eq!(blocks[0].kind, BlockKind::Heading(1));
        assert_eq!(blocks[1].kind, BlockKind::Heading(2));
        assert!(blocks[2].spans.iter().any(|s| s.code && s.text == "code"));
        assert!(blocks[2].spans.iter().any(|s| s.bold && s.text == "bold"));
        assert_eq!(blocks[3].kind, BlockKind::List(Some(3)));
        assert_eq!(blocks[3].children.len(), 2);
        assert_eq!(
            blocks[3].children[1].children[0].kind,
            BlockKind::List(None)
        );
        let code = &blocks[4];
        assert_eq!(code.kind, BlockKind::Code("rust".into()));
        assert_eq!(code.text(), "fn main() { println!(\"hi\"); }\n");
        assert!(code.syntax.len() > 1);
        assert_eq!(code.syntax.first().unwrap().range.start, 0);
        assert_eq!(code.syntax.last().unwrap().range.end, code.text().len());
        for mode in [Appearance::Light, Appearance::Dark] {
            assert!(code
                .syntax
                .iter()
                .all(|run| contrast(run.color(mode), mode.palette().background) >= 4.5));
        }
        assert!(code
            .syntax
            .iter()
            .any(|run| run.color(Appearance::Light) != run.color(Appearance::Dark)));
    }
    #[test]
    fn incomplete_stream_reparses_to_closed_fence() {
        let mut doc = Markdown::from("```python\nprint(");
        doc.push_str("'hello')\n```\n\nafter");
        assert_eq!(doc.root.children.len(), 2);
        assert_eq!(doc.root.children[0].text(), "print('hello')\n");
    }
}
