// Grava as cores e o nome do tema no PPTX gerado pelo pptxgenjs (que não escreve cores de tema).
// Uso: const { applyTheme } = require("./tema.js"); await applyTheme("deck.pptx", THEME);
// THEME = { name, colors: { dk1, lt1, dk2, lt2, accent1..accent6, hlink, folHlink } } com hex de 6 dígitos sem "#".
const fs = require("fs");
const JSZip = require("jszip");

const POSICOES = ["dk1", "lt1", "dk2", "lt2", "accent1", "accent2", "accent3", "accent4", "accent5", "accent6", "hlink", "folHlink"];
const escaparXml = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");

async function applyTheme(arquivo, tema) {
  const cores = tema.colors;
  for (const k of POSICOES) {
    if (!/^[0-9A-Fa-f]{6}$/.test(String(cores[k]))) throw new Error(`tema.colors.${k} = ${cores[k]}: use 6 dígitos hexadecimais, sem "#"`);
  }
  const zip = await JSZip.loadAsync(fs.readFileSync(arquivo));
  const parte = "ppt/theme/theme1.xml";
  if (!zip.file(parte)) throw new Error(`${arquivo} não tem ${parte}`);
  const nome = escaparXml(tema.name);
  const esquema = `<a:clrScheme name="${nome}">` + POSICOES.map(k => `<a:${k}><a:srgbClr val="${cores[k].toUpperCase()}"/></a:${k}>`).join("") + "</a:clrScheme>";
  const xmlTema = (await zip.file(parte).async("string"))
    .replace(/<a:clrScheme\b[\s\S]*?<\/a:clrScheme>/, () => esquema)
    .replace(/(<a:(?:theme|fontScheme)\b[^>]*?\bname=")[^"]*"/g, (_, inicio) => `${inicio}${nome}"`);
  if (!xmlTema.includes(esquema)) throw new Error(`${parte} sem <a:clrScheme> para substituir`);
  zip.file(parte, xmlTema);
  // O pptxgenjs grava <c:crossesAt val="autoZero"/>, que não é válido no esquema do PowerPoint; o correto é <c:crosses>.
  for (const n of Object.keys(zip.files).filter(n => /^ppt\/charts\/chart\d+\.xml$/.test(n))) {
    const xml = await zip.file(n).async("string");
    if (xml.includes('<c:crossesAt val="autoZero"/>')) zip.file(n, xml.replaceAll('<c:crossesAt val="autoZero"/>', '<c:crosses val="autoZero"/>'));
  }
  fs.writeFileSync(arquivo, await zip.generateAsync({ type: "nodebuffer", compression: "DEFLATE" }));
}

module.exports = { applyTheme };
