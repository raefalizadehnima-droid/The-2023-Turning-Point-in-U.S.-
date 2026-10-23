// Builds 01_Research_Manuscript.docx from 03_Processed_Data/manuscript_spec.json
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, ImageRun, AlignmentType,
  HeadingLevel, WidthType, BorderStyle, ShadingType, Footer, PageNumber, LevelFormat,
} = require("docx");

const root = path.resolve(__dirname, "..");
const spec = JSON.parse(fs.readFileSync(path.join(root, "03_Processed_Data", "manuscript_spec.json"), "utf8"));
const FONT = "Times New Roman";
const CONTENT_W = 9360; // US Letter, 1" margins

function pngSize(file) {
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}

const children = [];
let secNum = 0;

for (const b of spec) {
  if (b.type === "title") {
    children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 },
      children: [new TextRun({ text: b.text, bold: true, size: 30, font: FONT })] }));
  } else if (b.type === "h1") {
    children.push(new Paragraph({ heading: HeadingLevel.HEADING_1, spacing: { before: 280, after: 120 },
      children: [new TextRun({ text: b.text, bold: true, size: 26, font: FONT })] }));
  } else if (b.type === "h2") {
    children.push(new Paragraph({ heading: HeadingLevel.HEADING_2, spacing: { before: 200, after: 80 },
      children: [new TextRun({ text: b.text, bold: true, italics: true, size: 23, font: FONT })] }));
  } else if (b.type === "p") {
    children.push(new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 120, line: 300 },
      children: [new TextRun({ text: b.text, size: 22, font: FONT, italics: !!b.italic })] }));
  } else if (b.type === "ref") {
    children.push(new Paragraph({ spacing: { after: 80 }, indent: { left: 360, hanging: 360 },
      children: [new TextRun({ text: b.text, size: 20, font: FONT })] }));
  } else if (b.type === "figure") {
    const { w, h } = pngSize(b.path);
    const wpx = b.width || 600;
    const hpx = Math.round((wpx * h) / w);
    children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 60 }, keepNext: true,
      children: [new ImageRun({ type: "png", data: fs.readFileSync(b.path), transformation: { width: wpx, height: hpx },
        altText: { title: b.caption.slice(0, 40), description: b.caption.slice(0, 120), name: path.basename(b.path) } })] }));
    children.push(new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 200 },
      children: [new TextRun({ text: b.caption, size: 19, font: FONT })] }));
  } else if (b.type === "table") {
    children.push(new Paragraph({ spacing: { before: 160, after: 60 }, keepNext: true,
      children: [new TextRun({ text: b.title, bold: true, size: 20, font: FONT })] }));
    const ncol = b.header.length;
    // column widths proportional to sqrt of the longest cell, bounded
    const wts = b.header.map((h, j) => {
      const L = Math.max(h.length * 0.8, ...b.rows.map((r) => String(r[j]).length));
      return Math.min(Math.max(Math.sqrt(L), 2.2), 8);
    });
    const tot = wts.reduce((a, c) => a + c, 0);
    let widths = wts.map((w) => Math.floor((w / tot) * CONTENT_W));
    widths[widths.length - 1] += CONTENT_W - widths.reduce((a, c) => a + c, 0);
    const fs_ = ncol >= 9 ? 14 : ncol >= 7 ? 15 : 17;
    const border = { style: BorderStyle.SINGLE, size: 4, color: "999999" };
    const borders = { top: border, bottom: border, left: border, right: border };
    const mkCell = (txt, j, head) => new TableCell({
      width: { size: widths[j], type: WidthType.DXA }, borders,
      shading: head ? { fill: "E7E6E6", type: ShadingType.CLEAR, color: "auto" } : undefined,
      margins: { top: 40, bottom: 40, left: 60, right: 60 },
      children: [new Paragraph({ alignment: j === 0 ? AlignmentType.LEFT : AlignmentType.LEFT,
        children: [new TextRun({ text: String(txt), size: fs_, bold: head, font: FONT })] })],
    });
    const rows = [new TableRow({ tableHeader: true, children: b.header.map((h, j) => mkCell(h, j, true)) })];
    for (const r of b.rows) rows.push(new TableRow({ cantSplit: true, children: r.map((c, j) => mkCell(c, j, false)) }));
    children.push(new Table({ width: { size: CONTENT_W, type: WidthType.DXA }, columnWidths: widths, rows }));
    children.push(new Paragraph({ spacing: { after: 160 }, children: [] }));
  }
}

const doc = new Document({
  creator: "Authors to be completed",
  title: "Shifting Drug Patterns in U.S. Overdose Deaths, 2020-2024",
  styles: {
    default: { document: { run: { font: FONT, size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 26, bold: true, font: FONT }, paragraph: { spacing: { before: 280, after: 120 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 23, bold: true, italics: true, font: FONT }, paragraph: { spacing: { before: 200, after: 80 }, outlineLevel: 1 } },
    ],
  },
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 }, margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [new TextRun({ children: [PageNumber.CURRENT], size: 18, font: FONT })] })] }) },
    children,
  }],
});

const out = path.join(root, "01_Research_Manuscript.docx");
Packer.toBuffer(doc).then((buf) => { fs.writeFileSync(out, buf); console.log("wrote", out, buf.length); });
