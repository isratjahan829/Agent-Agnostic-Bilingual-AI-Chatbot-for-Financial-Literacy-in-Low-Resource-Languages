const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, AlignmentType, HeadingLevel,
  Table, TableRow, TableCell, WidthType, ShadingType, BorderStyle,
} = require("docx");

const FONT = "Times New Roman";
const SIZE = 22;          // half-points => 11pt
const content = JSON.parse(fs.readFileSync(process.argv[2], "utf8")).blocks;

const rules = (text) => {
  // "**bold**" and "*italic*" inline markers -> runs
  const out = [];
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)/g;
  let last = 0, m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push({ t: text.slice(last, m.index) });
    const token = m[0];
    if (token.startsWith("**")) out.push({ t: token.slice(2, -2), bold: true });
    else if (token.startsWith("`")) out.push({ t: token.slice(1, -1), mono: true });
    else out.push({ t: token.slice(1, -1), italics: true });
    last = m.index + token.length;
  }
  if (last < text.length) out.push({ t: text.slice(last) });
  return out;
};

const runs = (text, extra = {}) =>
  rules(text).map((piece) => new TextRun({
    text: piece.t,
    bold: piece.bold || extra.bold,
    italics: piece.italics || extra.italics,
    font: piece.mono ? "Consolas" : FONT,
    size: piece.mono ? SIZE - 3 : (extra.size || SIZE),
    color: extra.color,
  }));

const para = (text, opts = {}) => new Paragraph({
  children: runs(text || "", opts),
  alignment: opts.alignment,
  spacing: { after: opts.after === undefined ? 120 : opts.after, line: 276 },
  indent: opts.indent,
  border: opts.border,
  heading: opts.heading,
});

const TABLE_WIDTH = 9360;

function makeTable(rows) {
  const columns = rows[0].length;
  const widths = Array(columns).fill(Math.floor(TABLE_WIDTH / columns));
  widths[0] += TABLE_WIDTH - widths.reduce((a, b) => a + b, 0);
  return new Table({
    columnWidths: widths,
    width: { size: TABLE_WIDTH, type: WidthType.DXA },
    rows: rows.map((cells, r) => new TableRow({
      tableHeader: r === 0,
      children: cells.map((cell, c) => new TableCell({
        width: { size: widths[c], type: WidthType.DXA },
        shading: r === 0 ? { type: ShadingType.CLEAR, fill: "E8EDF3" } : undefined,
        margins: { top: 60, bottom: 60, left: 90, right: 90 },
        children: [new Paragraph({
          children: runs(String(cell), { bold: r === 0, size: SIZE - 2 }),
          spacing: { after: 0, line: 240 },
        })],
      })),
    })),
  });
}

const children = [];
for (const block of content) {
  switch (block.type) {
    case "title":
      children.push(para(block.text, { bold: true, size: SIZE + 4, after: 200 }));
      break;
    case "p":
      children.push(para(block.text, { after: block.after }));
      break;
    case "center":
      children.push(para(block.text, { alignment: AlignmentType.CENTER }));
      break;
    case "reviewer":
      children.push(new Paragraph({ children: [], spacing: { after: 200 } }));
      children.push(para(block.text, { bold: true, size: SIZE + 2, after: 40 }));
      children.push(new Paragraph({
        children: [],
        spacing: { after: 120 },
        border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: "333333" } },
      }));
      break;
    case "comment":
      children.push(para(block.text, { bold: true, after: 60 }));
      break;
    case "quote":
      children.push(new Paragraph({
        children: runs(block.text, { italics: true }),
        indent: { left: 720, right: 360 },
        spacing: { after: 120, line: 276 },
        border: { left: { style: BorderStyle.SINGLE, size: 6, color: "7F9DB9", space: 8 } },
      }));
      break;
    case "code":
      children.push(new Paragraph({
        children: [new TextRun({ text: block.text, font: "Consolas", size: SIZE - 4 })],
        indent: { left: 720 },
        spacing: { after: 80, line: 240 },
      }));
      break;
    case "bullet":
      children.push(new Paragraph({
        children: runs(block.text),
        bullet: { level: 0 },
        spacing: { after: 80, line: 276 },
      }));
      break;
    case "table":
      children.push(makeTable(block.rows));
      children.push(new Paragraph({ children: [], spacing: { after: 160 } }));
      break;
    case "spacer":
      children.push(new Paragraph({ children: [], spacing: { after: 200 } }));
      break;
    default:
      throw new Error("unknown block type " + block.type);
  }
}

const doc = new Document({
  styles: { default: { document: { run: { font: FONT, size: SIZE } } } },
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 },
                          margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } } },
    children,
  }],
});

Packer.toBuffer(doc).then((buffer) => {
  fs.writeFileSync(process.argv[3], buffer);
  console.log("wrote", process.argv[3], buffer.length, "bytes");
});
