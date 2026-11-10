// Apresentação executiva — Coffee Sales (PT-BR), ~30 minutos: 16 slides principais + apêndice técnico indexado (A1–A12).
// A10–A12 guardam o detalhe estatístico que saiu dos slides principais 6, 9 e 11.
// Nenhum número é digitado à mão: todos vêm de dados_deck.json, que dados_deck.py extrai das tabelas de reports/.
// Uso (nesta pasta): python dados_deck.py && npm ci && node gerar_deck.js [saida.pptx]
const pptxgen = require("pptxgenjs");
const JSZip = require("jszip");
const fs = require("fs");
const path = require("path");
const { applyTheme } = require("./tema.js");   // cores do tema (o pptxgenjs não as grava)
const D = require("./dados_deck.json");
const SAIDA = process.argv[2] || "../coffee_sales_executiva.pptx";

const THEME = {
  name: "Coffee Sales", headFontFace: "Cambria", bodyFontFace: "Calibri",
  colors: { dk1: "1F1A17", lt1: "FFFFFF", dk2: "3B2A20", lt2: "F2F1EF", accent1: "B8652A", accent2: "2A78D6",
            accent3: "EB6834", accent4: "1BAF7A", accent5: "7A5C45", accent6: "8A8A8A", hlink: "2A78D6", folHlink: "7A5C45" },
};
const HEX = THEME.colors;
const COR = { m1: "2A78D6", m2: "EB6834", total: "1BAF7A", contexto: "A3A29C", eixo: "52514E", grade: "E6E3DF", linha: "C9C8C2" };
const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";            // 13,33" × 7,5"
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
pres.title = "Coffee Sales — demanda, produtos e previsão por máquina";
pres.author = "Matheus Felix";
const C = pres.SchemeColor;
const W = 13.333, M = 0.6;

// Posição de cada slide no deck final (para os links internos). Principal: 1–16; índice do apêndice: 17; A1–A12: 18–29.
const N = { indice: 17, A1: 18, A2: 19, A3: 20, A4: 21, A5: 22, A6: 23, A7: 24, A8: 25, A9: 26, A10: 27, A11: 28, A12: 29 };

// ------------------------------------------------------------------ layouts (idênticos à v1)
pres.defineSlideMaster({
  title: "CAPA", background: { color: HEX.dk2 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: M, y: 2.1, w: 11.5, h: 1.9, fontFace: "Cambria", fontSize: 44, bold: true, color: C.background1, valign: "bottom", align: "left" }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: M, y: 4.2, w: 11.5, h: 1.2, fontFace: "Calibri", fontSize: 20, color: C.background2, valign: "top" }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTEUDO", background: { color: HEX.lt1 },
  objects: [
    { placeholder: { options: { name: "kicker", type: "body", x: M, y: 0.35, w: 9, h: 0.35, fontFace: "Calibri", fontSize: 12, bold: true, color: C.accent1, valign: "middle" }, text: "" } },
    { placeholder: { options: { name: "title", type: "title", x: M, y: 0.7, w: W - 2 * M, h: 0.85, fontFace: "Cambria", fontSize: 28, bold: true, color: C.text2, valign: "top", align: "left" }, text: "" } },
    { text: { text: "Coffee Sales · demanda, produtos e previsão por máquina", options: { x: M, y: 7.05, w: 7, h: 0.3, fontFace: "Calibri", fontSize: 10, color: C.accent6 } } },
  ],
  slideNumber: { x: 12.3, y: 7.05, w: 0.5, h: 0.3, fontFace: "Calibri", fontSize: 10, color: C.accent6, align: "right" },
});

// ------------------------------------------------------------------ utilitários (componentes visuais + links internos)
function img(slide, chave, x, y, w, h, alinhar = "centro") {
  const im = D.imagens[chave];
  let ww = w, hh = w / im.ratio;
  if (hh > h) { hh = h; ww = h * im.ratio; }
  const xx = alinhar === "esquerda" ? x : x + (w - ww) / 2;
  const yy = alinhar === "esquerda" ? y : y + (h - hh) / 2;
  slide.addImage({ path: path.resolve(__dirname, im.path), x: xx, y: yy, w: ww, h: hh, altText: chave, objectName: `img_${chave}` });
}
// Cartão de número-destaque: número no topo, rótulo logo abaixo (altura do número calculada pelo corpo da fonte).
function card(slide, x, y, w, h, numero, rotulo, o = {}) {
  const tam = o.tamanho || 30, numH = tam / 72 * 1.25 + 0.06;
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.12, fill: { color: o.escuro ? C.text2 : C.background2 }, line: { type: "none" }, objectName: "card" });
  slide.addText(numero, { x: x + 0.2, y: y + 0.12, w: w - 0.4, h: numH, fontFace: "Cambria", fontSize: tam, bold: true, color: o.escuro ? C.background1 : C.accent1, valign: "middle", margin: 0, isTextBox: true, fit: "none" });
  slide.addText(rotulo, { x: x + 0.2, y: y + 0.14 + numH, w: w - 0.4, h: h - numH - 0.26, fontFace: "Calibri", fontSize: o.rotulo || 13, color: o.escuro ? C.background2 : C.text1, valign: "top", margin: 0, isTextBox: true });
}
function caixaTexto(slide, titulo, itens, x, y, w, h, o = {}) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.1, fill: { color: o.fundo || C.background2 }, line: { type: "none" }, objectName: "caixa" });
  const runs = [{ text: titulo, options: { bold: true, fontSize: o.tituloTam || 16, color: o.corTitulo || C.text2, breakLine: true, paraSpaceAfter: 6 } }];
  itens.forEach((t, i) => runs.push({ text: t, options: { bullet: o.semMarcador ? false : true, fontSize: o.tam || 14, color: o.corTexto || C.text1, breakLine: i < itens.length - 1, paraSpaceAfter: o.espaco || 5 } }));
  slide.addText(runs, { x: x + 0.15, y: y + 0.12, w: w - 0.3, h: h - 0.24, valign: "top", fontFace: "Calibri", margin: 2, isTextBox: true });
}
function legenda(slide, texto, x, y, w) {
  slide.addText(texto, { x, y, w, h: 0.3, fontFace: "Calibri", fontSize: 12, color: C.accent6, margin: 0, valign: "middle", isTextBox: true });
}
function circulo(slide, x, y, d, cor, texto) {
  slide.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: cor }, line: { type: "none" }, objectName: "circulo" });
  if (texto) slide.addText(texto, { x, y, w: d, h: d, align: "center", valign: "middle", fontFace: "Cambria", fontSize: d > 0.55 ? 20 : 16, bold: true, color: C.background1, margin: 0, isTextBox: true });
}
function marcador(slide, x, y, d, fundo, corTexto, texto, tam) {
  slide.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: fundo }, line: { type: "none" }, objectName: "marcador" });
  slide.addText(texto, { x, y, w: d, h: d, align: "center", valign: "middle", fontFace: "Cambria", fontSize: tam, bold: true, color: corTexto, margin: 0, isTextBox: true });
}
function conteudo(secao, kicker, titulo) {
  const s = pres.addSlide({ masterName: "CONTEUDO", sectionTitle: secao });
  s.addText(kicker, { placeholder: "kicker" });
  s.addText(titulo, { placeholder: "title" });
  return s;
}
// Área clicável invisível (forma transparente por cima do texto): leva a outro slide no modo apresentação sem pintar o texto de azul.
function link(slide, x, y, w, h, destino, dica) {
  slide.addShape(pres.shapes.RECTANGLE, { x, y, w, h, fill: { color: "FFFFFF", transparency: 100 }, line: { type: "none" },
    hyperlink: { slide: destino, tooltip: dica }, objectName: `link_slide_${destino}` });
}
// Rodapé dos slides principais: aponta o detalhe técnico no apêndice (A1–A9), com link.
function rodape(slide, refs) {
  let xr = 12.1;
  [...refs].reverse().forEach(([sig, rot]) => {
    const w = 0.16 + (sig.length + 1 + rot.length) * 0.064;
    const x = xr - w;
    slide.addText([{ text: sig, options: { bold: true, color: C.accent1 } }, { text: ` ${rot}`, options: { color: C.accent6 } }],
      { x, y: 7.05, w, h: 0.3, fontFace: "Calibri", fontSize: 10, align: "right", valign: "middle", margin: 0, isTextBox: true });
    link(slide, x, 7.05, w, 0.3, N[sig], `Apêndice ${sig}: ${rot}`);
    xr = x - 0.16;
  });
  slide.addText("Detalhe no apêndice:", { x: xr - 1.3, y: 7.05, w: 1.3, h: 0.3, fontFace: "Calibri", fontSize: 10, color: C.accent6, align: "right", valign: "middle", margin: 0, isTextBox: true });
}
// Slides do apêndice: kicker com a sigla e link de volta ao índice.
function apendice(sig, nome, titulo) {
  const s = conteudo("Apêndice técnico", `APÊNDICE ${sig} · ${nome}`, titulo);
  s.addText("← Índice do apêndice", { x: 9.9, y: 7.05, w: 2.2, h: 0.3, fontFace: "Calibri", fontSize: 10, bold: true, color: C.accent1, align: "right", valign: "middle", margin: 0, isTextBox: true });
  link(s, 9.9, 7.05, 2.2, 0.3, N.indice, "Voltar ao índice do apêndice");
  return s;
}
// Opções de eixo comuns a todos os gráficos nativos (objeto novo a cada chamada: o pptxgenjs altera as opções).
function eixos(extra) {
  return Object.assign({
    showTitle: false, catAxisLabelColor: COR.eixo, valAxisLabelColor: COR.eixo, catAxisLabelFontSize: 11, valAxisLabelFontSize: 11,
    catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", catAxisLineShow: true, catAxisLineColor: COR.linha, valAxisLineShow: false,
    valGridLine: { color: COR.grade, size: 0.75 }, catGridLine: { style: "none" }, dataLabelColor: HEX.dk2, dataLabelFontSize: 11,
    dataLabelFontFace: "+mn-lt", legendFontFace: "+mn-lt", legendFontSize: 12, legendColor: HEX.dk2,
  }, extra);
}
const UM = " u.m.";
const fmt = (v, d = 1) => (v === null || v === undefined) ? "—" : Number(v).toFixed(d).replace(".", ",");
const mil = (v) => Math.round(v).toLocaleString("pt-BR");
const sinal = (v, d = 0) => `${v > 0 ? "+" : v < 0 ? "−" : ""}${fmt(Math.abs(v), d)}%`;
const pct0 = (v) => `${fmt(100 * v, 0)}%`;
const dataBR = (iso) => `${iso.slice(8, 10)}/${iso.slice(5, 7)}/${iso.slice(0, 4)}`;
const esc = (s, a) => D.escolhidos.find(r => r.serie === s && r.alvo === a);
const TS = (alvo, ab) => D.total_soma.find(r => r.alvo === alvo && r.abordagem === ab);
const ganhoSoma = (alvo) => Math.round(100 * (1 - TS(alvo, "Soma M1 + M2").MAE / TS(alvo, "Total direto").MAE));

// ------------------------------------------------------------------ dados
const F = D.futuro, f1 = F.machine_1, f2 = F.machine_2, ft = F.total, fs_ = F.soma;
const m1d = esc("Máquina 1", "demanda"), m2d = esc("Máquina 2", "demanda");
const SEM = D.semanal_m1, semAntes = SEM.testes[0], semDepois = SEM.testes[1];
const TP = D.ticket_publico, TR = D.ticket_resumo;
const CC = D.concentracao, CM2 = CC["Máquina 2"], CTOT = CC["Total"];
const FAM = (s, f) => D.familias[`${s}|${f}`];
const AQ = D.anomalias.queda, AP = D.anomalias.pico;
const descMax = Math.min(...Object.values(D.desconto_m2));
const bt_ganho_m1 = Math.round(100 * (1 - m1d.mase_bt));
const periodoFut = `${F.machine_1.demanda.inicio.slice(8, 10)}/${F.machine_1.demanda.inicio.slice(5, 7)}–${F.machine_1.demanda.fim.slice(8, 10)}/${F.machine_1.demanda.fim.slice(5, 7)}/${F.machine_1.demanda.fim.slice(0, 4)}`;
const HR = D.horario;
const TD = D.tendencia["Máquina 1"], TDT = D.tendencia["Total"], SR = D.semanal_regimes, DG = D.degraus_m1;
const PG = D.periodograma_m1_2025, F25 = D.fisher_2025;
const AT = D.atributos, IM = D.importancia, EV = D.evidencia, QL = D.qualidade, DU = D.duplicados;
const ql = (serie, alvo, modelo) => QL.find(r => r["série"] === serie && r.alvo === alvo && (!modelo || r.modelo === modelo));
const sd = ql("Total", "demanda", "Soma M1 + M2"), sx = ql("Total", "demanda", "SARIMAX");
const q1 = ql("Máquina 1", "demanda"), q2 = ql("Máquina 2", "demanda");

// ================================================================== 1–2. capa e resumo
pres.addSection({ title: "Abertura" });
let s = pres.addSlide({ masterName: "CAPA", sectionTitle: "Abertura" });
s.addText("Coffee Sales: demanda, produtos e previsão por máquina", { placeholder: "title" });
s.addText("O que os dados de vendas dizem sobre clientes, cardápio e operação · 2 máquinas · mar/2024 a mar/2025", { placeholder: "body" });
s.addNotes("[~0,5 min] Apresentação de ~30 minutos em 16 slides. Roteiro: qual é o problema de negócio; o que os dados mostram sobre as máquinas, os produtos e os clientes; quais mudanças estruturais aconteceram; se houve canibalização; como os modelos foram construídos e validados; por que os escolhidos; as previsões e sua incerteza; e as decisões de negócio. Os detalhes técnicos ficam no apêndice (índice no slide 17, com links para cada slide) e no repositório case_coffee_sales. Tempos sugeridos no início de cada nota (total ~27 min). Onde cada requisito do desafio aparece: (1) contexto, dor e impacto: slide 3; (2) alternativas e trade-offs: slides 10 e 11; (3) dados, preparação, atributos e decisões metodológicas: slides 3 a 10; (4) métricas, previsões, implicações, aprendizados e limitações: slides 11, 12, 13 e 16; (5) arquitetura para produção: slide 14; (6) limitações, riscos e próximos passos: slide 15.");

s = conteudo("Abertura", "RESUMO EXECUTIVO", "Demanda nova, ticket menor e receita em poucos produtos");
const kpis = [
  ["2,8×", "vendas/dia em fev–mar/2025 (20,4) frente ao fim de 2024 (7,3): nova clientela na M1 + abertura da M2"],
  [`${CTOT["produtos para 80% da receita"]} de ${CTOT["produtos"]}`, `produtos fazem 80% da receita; ${CM2["produtos com ≤ 5 vendas"]} dos ${CM2["produtos"]} itens da M2 venderam até 5 vezes em 44 dias`],
  [sinal(TR.m2_vs_m1), `ticket da M2 frente à M1; na M1, o público de 2025 gasta ${fmt(Math.abs(TR.var_m1))}% menos por compra`],
  ["2 públicos", `M1 vende em dias úteis (fim de semana ${sinal(semDepois["fim de semana vs dias úteis (%)"])} em 2025); a M2 faz ${fmt(D.fds_m2_pct, 0)}% das vendas no fim de semana`],
];
kpis.forEach((k, i) => card(s, M + i * 3.07, 1.7, 2.9, 1.6, k[0], k[1], { tamanho: 28, rotulo: 12 }));
caixaTexto(s, "O que recomendamos", [
  `Revisar a cauda da M2 (${CM2["produtos com ≤ 5 vendas"]} itens, ${fmt(CM2["% receita da cauda"], 0)}% da receita), começando por famílias inteiras de insumo, como baunilha`,
  `Garantir o essencial: leite (${fmt(FAM("Máquina 1", "leite").pct_vendas, 0)}% das vendas da M1) e os ${CTOT["produtos para 80% da receita"]} produtos que fazem 80% da receita`,
  `Recuperar ticket: oferta com leite sobre o Americano e revisão da tabela da M2 (até ${fmt(Math.abs(descMax), 0)}% abaixo da M1)`,
  "Abastecer por dia da semana: a M1 antes de seg/qua/sex e a M2 antes do fim de semana",
], M, 3.5, 7.45, 3.35, { tam: 14, tituloTam: 17, espaco: 9 });
caixaTexto(s, "Como prever os próximos 14 dias", [
  "Um modelo por máquina; o Total é a soma das duas",
  `M1 · Holt-Winters: ${fmt(f1.demanda.media)} vendas/dia`,
  `M2 · Random Forest: ${fmt(f2.demanda.media)} vendas/dia`,
  `Total: ${fmt(fs_.demanda.media)} vendas e ${mil(fs_.receita.media)}${UM} por dia`,
  `Erro típico: ~${fmt(m1d.wape, 0)}% na M1; ~${fmt(m2d.wape, 0)}% na M2, que tem só 44 dias de histórico`,
], M + 7.65, 3.5, W - 2 * M - 7.65, 3.35, { tam: 14, tituloTam: 17, fundo: C.text2, corTitulo: C.background1, corTexto: C.background1, espaco: 9 });
s.addNotes("[~2 min] Mensagem principal: em 2025 a operação ganhou um público novo, que compra mais vezes, mas gasta menos por compra e se concentra em poucos produtos. Isso abre três frentes de negócio: enxugar o cardápio da M2, proteger o estoque do essencial e recuperar ticket. A previsão por máquina dá o volume esperado para planejar insumos e serve de base de comparação para medir essas ações.");

// ================================================================== 3. contexto + definição do problema de previsão
pres.addSection({ title: "1. Contexto e problema" });
s = conteudo("1. Contexto e problema", "1 · CONTEXTO E PROBLEMA DE PREVISÃO", "Prever 14 dias à frente apoia reposição, roteiro e caixa");
s.addText([{ text: "Objetivo: ", options: { bold: true, color: C.accent1 } },
           { text: "prever demanda e faturamento por máquina e no total, 14 dias à frente, e tirar dos mesmos dados conclusões sobre público e cardápio", options: { color: C.text2 } }],
  { x: M, y: 1.58, w: W - 2 * M, h: 0.38, fontFace: "Calibri", fontSize: 14, valign: "middle", margin: 0, isTextBox: true });
// esquerda: a oportunidade (decisões que a previsão apoia) e o impacto esperado
s.addText("Oportunidade: o que a previsão apoia", { x: M, y: 2.1, w: 5.65, h: 0.32, fontFace: "Cambria", fontSize: 15, bold: true, color: C.text2, valign: "middle", margin: 0, isTextBox: true });
const decisoes = [
  ["Reposição de insumos", "volume por máquina e dia da semana; o limite superior do intervalo dimensiona o estoque de segurança"],
  ["Roteiro de abastecimento", `a M1 vende em dias úteis; a M2 concentra ${fmt(D.fds_m2_pct, 0)}% no fim de semana`],
  ["Cardápio e preço", "quais produtos sustentam a receita e quais ocupam espaço e insumo sem vender"],
  ["Monitoramento", "um dia fora do intervalo de 95% sinaliza falha, fechamento ou mudança de público"],
];
decisoes.forEach((d, i) => {
  const y = 2.5 + i * 0.78;
  circulo(s, M, y + 0.12, 0.45, C.accent1, String(i + 1));
  s.addText([{ text: d[0], options: { bold: true, fontSize: 14, color: C.text2, breakLine: true } }, { text: d[1], options: { fontSize: 12, color: C.text1 } }],
    { x: M + 0.62, y, w: 5.03, h: 0.7, valign: "middle", fontFace: "Calibri", margin: 0, isTextBox: true });
});
s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: M, y: 5.68, w: 5.65, h: 1.22, rectRadius: 0.1, fill: { color: C.background2 }, line: { type: "none" }, objectName: "impacto" });
s.addText([{ text: "Impacto esperado: ", options: { bold: true, color: C.accent1 } },
           { text: "menos insumo parado ou vencido, menos ruptura do essencial, ticket recuperado e abastecimento por dia da semana; a previsão é a base para medir cada piloto.", options: { color: C.text1, breakLine: true } },
           { text: "Sem custos, margens, estoque, rupturas, horário oficial, local ou país nos dados: ", options: { bold: true, color: C.text2 } },
           { text: "nenhum ganho financeiro é estimado.", options: { color: C.text1 } }],
  { x: M + 0.18, y: 5.74, w: 5.29, h: 1.1, fontFace: "Calibri", fontSize: 12, valign: "middle", margin: 0, isTextBox: true });
// direita: o que exatamente é previsto (unidade da série, alvos, segmentação, horizonte, histórico)
const xT = 6.55, wT = W - M - xT;
s.addText("O que é previsto", { x: xT, y: 2.1, w: wT, h: 0.32, fontFace: "Cambria", fontSize: 15, bold: true, color: C.text2, valign: "middle", margin: 0, isTextBox: true });
const lp = (a, b) => [{ text: a, options: { bold: true, fontSize: 12, color: HEX.dk2, fill: { color: HEX.lt2 } } }, { text: b, options: { fontSize: 11.5 } }];
s.addTable([
  lp("Observação", "1 dia de 1 máquina, em calendário contínuo; dia sem registro = 0 vendas e 0 de receita (não é imputado)"),
  lp("Alvo 1 · vendas", "nº de registros de venda no dia: 1 registro = 1 produto vendido e seu valor (os gráficos o chamam de \"transação\")"),
  lp("Registro ≠ pedido", "não há ID de pedido: um pedido com vários itens vira vários registros (na M2, há itens diferentes no mesmo segundo)"),
  lp("Alvo 2 · receita", "soma dos valores dos registros do dia, em u.m."),
  lp("Frequência", "diária: há menos de 1 venda por hora"),
  lp("Séries", "M1 e M2 previstas separadamente; Total = soma das duas previsões (antes de 08/02/2025, só a M1)"),
  lp("Histórico", "M1: 01/03/2024–23/03/2025 · 388 dias · 3.636 registros (9,4/dia)\nM2: 08/02/2025–23/03/2025 · 44 dias · 262 registros (~6/dia)"),
  lp("Horizonte", "14 dias (24/03–06/04/2025); avaliação no teste com os últimos 14 dias observados (10–23/03/2025)"),
], { x: xT, y: 2.5, w: wT, colW: [1.55, wT - 1.55], fontFace: "Calibri", color: HEX.dk1, border: { type: "solid", pt: 0.5, color: "D9D6D2" }, valign: "middle", margin: [3, 6, 3, 6] });
rodape(s, [["A1", "Qualidade dos dados"]]);
s.addNotes("[~2 min] O problema é prever demanda e faturamento por máquina e no total, em horizonte de 14 dias, e tirar dos mesmos dados conclusões sobre público e cardápio. A unidade da série é o dia de cada máquina. 'Vendas' é a contagem de registros do log (cada registro é um produto vendido, com seu valor); como não há ID de pedido, registros não equivalem a pedidos: na M2 há itens diferentes no mesmo segundo, típico de pedido com vários itens, e na M1 recompras do mesmo item em ~1 minuto parecem duas unidades na mesma compra. Nada é removido; dias sem registro dentro do período ativo valem zero (7 na M1, 3 na M2), e a M2 não existe antes de 08/02/2025 (não é zero nem ausente). O Total previsto é a soma das previsões por máquina; o modelo único do Total fica só como checagem. Não inventamos custos: sem margem e estoque, o impacto é descrito em volume, ticket e participação. Detalhes de ausentes, zeros, outliers e duplicados no apêndice A1.");

// ================================================================== 4–5. máquinas, clientes e produtos
pres.addSection({ title: "2. Máquinas, clientes e produtos" });
s = conteudo("2. Máquinas, clientes e produtos", "2 · DUAS OPERAÇÕES", "index_2 é uma segunda máquina, não uma cópia de index_1");
[["0 de 262", "vendas da M2 com par na M1 (mesmo produto, ±60 s)"], ["5%", "cópia plantada detectada pelo controle positivo"],
 ["55%", "das vendas da M2 são produtos que a M1 nunca vendeu"], ["31% × 0%", "de pagamentos em dinheiro (M2 × M1)"]].forEach((c, i) =>
  card(s, M, 1.75 + i * 1.28, 3.6, 1.15, c[0], c[1], { tamanho: 24, rotulo: 12 }));
img(s, "perfis", 4.5, 1.75, 8.2, 3.6);
s.addText("Volumes diários sem correlação (ρ = −0,11) e perfis horário e semanal próprios: são dois pontos de venda com públicos diferentes. Somar os dois no Total não conta a mesma venda duas vezes.",
  { x: 4.5, y: 5.55, w: 8.2, h: 1.0, fontFace: "Calibri", fontSize: 14, color: C.text1, valign: "top", isTextBox: true });
s.addNotes("[~1,5 min] Testes reaproveitados do notebook teste_index2_nova_maquina, sem mudança de metodologia. O controle positivo mostra que os testes detectariam uma cópia, então o resultado negativo é informativo. Para o negócio: a M2 tem cardápio, preço e público próprios.");

s = conteudo("2. Máquinas, clientes e produtos", "2 · PRODUTOS", `${CTOT["produtos para 80% da receita"]} de ${CTOT["produtos"]} produtos fazem 80% da receita`);
legenda(s, "Participação na receita total, mar/2024–mar/2025", M, 1.62, 7.4);
s.addChart(pres.charts.BAR, [{ name: "% da receita", labels: D.receita_produtos_total.produtos, values: D.receita_produtos_total.pct.map(v => v / 100) }],
  eixos({ x: M, y: 1.95, w: 7.4, h: 4.95, barDir: "bar", chartColors: [HEX.accent1], barGapWidthPct: 45, showValue: true, dataLabelPosition: "outEnd",
          dataLabelFormatCode: "0%", catAxisOrientation: "maxMin", valAxisHidden: true, valGridLine: { style: "none" }, showLegend: false,
          catAxisLabelFontSize: 12, valAxisMaxVal: 0.28, valAxisMinVal: 0 }));
card(s, 8.3, 1.7, W - M - 8.3, 1.45, `${fmt(CTOT["% receita dos 3 maiores"], 0)}%`, "da receita vem de só 3 produtos: Latte, Americano with milk e Cappuccino", { tamanho: 26, rotulo: 13 });
caixaTexto(s, "Máquina 2: cardápio amplo, cauda longa", [
  `${CM2["produtos"]} produtos; ${CM2["produtos para 80% da receita"]} fazem 80% da receita`,
  `${CM2["produtos com ≤ 5 vendas"]} venderam até 5 vezes em 44 dias e somam ${fmt(CM2["% receita da cauda"], 0)}% da receita`,
  `baunilha: ${FAM("Máquina 2", "baunilha").produtos} produtos e ${FAM("Máquina 2", "baunilha").vendas} vendas (${fmt(FAM("Máquina 2", "baunilha").pct_receita, 0)}% da receita)`,
  `whiskey (aroma): ${FAM("Máquina 2", "whiskey (aroma)").produtos} produtos e ${fmt(FAM("Máquina 2", "whiskey (aroma)").pct_receita, 0)}% da receita, o diferencial da M2`,
  `leite está em ${fmt(FAM("Máquina 1", "leite").pct_vendas, 0)}% das vendas da M1 e ${fmt(FAM("Máquina 2", "leite").pct_vendas, 0)}% da M2`,
], 8.3, 3.35, W - M - 8.3, 3.5, { tam: 14, tituloTam: 16 });
s.addNotes("[~1,5 min] Os 8 produtos do cardápio da M1 fazem 97% da receita total. A M2 tem 30 produtos, e a maioria vende menos de uma vez por semana. As famílias de insumo vêm do nome do produto (a base não traz fichas técnicas): cortar só parte de uma família não tira o insumo de linha. Whiskey não é cauda: é 32% da receita da M2 e não existe na M1.");

// ================================================================== 6–8. mudanças estruturais
pres.addSection({ title: "3. Mudanças estruturais" });
// --- 6. horário + evolução da demanda: degraus numerados no gráfico, efeito do horário em destaque, conclusão de negócio.
// Os testes (ΔBIC, Sup-Wald, IRR e p-valores) ficam em A2, A3 e A10.
s = conteudo("3. Mudanças estruturais", "3 · HORÁRIO E EVOLUÇÃO DA DEMANDA", "Demanda em degraus: o primeiro veio da ampliação do horário");
legenda(s, `Máquina 1 · demanda média semanal (cinza), patamar de cada regime (azul) e reta de tendência (tracejado) · treino até ${dataBR(TD.fim)}`, M, 1.6, W - 2 * M);
const g6 = { x: M, y: 1.9, w: W - 2 * M, h: 2.65, lx: 0.04, ly: 0.12, lw: 0.95, lh: 0.74 };
s.addChart([
  { type: pres.charts.LINE, data: [{ name: "Média semanal", labels: DG.rotulos, values: DG.demanda }, { name: "Nível do regime (degraus)", labels: DG.rotulos, values: DG.nivel }],
    options: { chartColors: [COR.contexto, COR.m1], lineSize: 2, lineDataSymbol: "none" } },
  { type: pres.charts.LINE, data: [{ name: "Tendência linear", labels: DG.rotulos, values: DG.reta }],
    options: { chartColors: [HEX.accent1], lineSize: 2, lineDash: "dash", lineDataSymbol: "none" } },
], eixos({ x: g6.x, y: g6.y, w: g6.w, h: g6.h, layout: { x: g6.lx, y: g6.ly, w: g6.lw, h: g6.lh }, valAxisMinVal: 0, valAxisMaxVal: 20, valAxisMajorUnit: 5,
           valAxisLabelFormatCode: "0", catAxisLabelFrequency: 13, showLegend: false }));
// semana (rótulo = domingo de fechamento, resample "W") que contém cada quebra de A3
const semanaDe = (iso) => DG.rotulos.findIndex(r => new Date(`20${r.slice(6, 8)}-${r.slice(3, 5)}-${r.slice(0, 2)}`) >= new Date(iso));
const degraus = [
  { data: "2024-05-14", tema: "horário", corpo: [{ text: sinal(100 * (HR.irr_total - 1)), options: { bold: true, fontSize: 15, fontFace: "Cambria" } },
      { text: ` de demanda total, núcleo 10–19h estável; vendas fora de 10–19h: ${fmt(HR.fora_antes, 0)}% → ${fmt(HR.fora_depois, 0)}% (janela inferida)` }] },
  { data: "2024-07-24", tema: "preço", corpo: [{ text: "Patamar sobe para 11,4/dia, logo após a redução de preços de julho" }] },
  { data: "2024-11-16", tema: "preço", corpo: [{ text: "Patamar cai para 7,3/dia, depois da alta de preços de outubro" }] },
  { data: "2025-01-30", tema: "público", corpo: [{ text: `Patamar sobe para 14,4/dia com a nova clientela; vendas após 19h: ${fmt(TP["M1 · base anterior (mesma tabela)"].noite, 0)}% → ${fmt(TP["M1 · desde a quebra (todos)"].noite, 0)}%` }] },
];
const yTopo = g6.y + g6.h * g6.ly, yBase = g6.y + g6.h * (g6.ly + g6.lh);
degraus.forEach((dg, i) => {
  const xm = g6.x + g6.w * (g6.lx + g6.lw * (semanaDe(dg.data) + 0.5) / DG.rotulos.length);
  s.addShape(pres.shapes.LINE, { x: xm, y: yTopo + 0.3, w: 0, h: yBase - yTopo - 0.3, line: { color: HEX.accent1, width: 1, dashType: "dash" }, objectName: `degrau_linha_${i + 1}` });
  marcador(s, xm - 0.16, yTopo - 0.04, 0.32, i === 0 ? C.accent1 : C.text2, C.background1, String(i + 1), 12);
});
const wD = (W - 2 * M - 3 * 0.16) / 4;
degraus.forEach((dg, i) => {
  const x = M + i * (wD + 0.16), y = 4.7, h = 1.4, destaque = i === 0;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: wD, h, rectRadius: 0.1, fill: { color: destaque ? C.accent1 : C.background2 }, line: { type: "none" }, objectName: `degrau_${i + 1}` });
  marcador(s, x + 0.15, y + 0.13, 0.32, destaque ? C.background1 : C.text2, destaque ? C.accent1 : C.background1, String(i + 1), 12);
  s.addText(`${dataBR(dg.data)} · ${dg.tema}`, { x: x + 0.56, y: y + 0.1, w: wD - 0.66, h: 0.38, fontFace: "Cambria", fontSize: 14, bold: true, color: destaque ? C.background1 : C.text2, valign: "middle", margin: 0, isTextBox: true });
  s.addText(dg.corpo.map(r => ({ text: r.text, options: Object.assign({ color: destaque ? C.background1 : C.text1 }, r.options || {}) })),
    { x: x + 0.15, y: y + 0.52, w: wD - 0.3, h: h - 0.6, fontFace: "Calibri", fontSize: 12, valign: "top", margin: 0, isTextBox: true });
});
s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: M, y: 6.22, w: W - 2 * M, h: 0.68, rectRadius: 0.1, fill: { color: C.text2 }, line: { type: "none" }, objectName: "conclusao_negocio" });
s.addText([{ text: "Para o negócio: ", options: { bold: true } },
           { text: "planejar pelo patamar atual, sem projetar crescimento, e detectar rápido quando ele muda. Cada degrau coincide com uma mudança de negócio (horário, preço, público): associação no tempo, não prova de causa." }],
  { x: M + 0.2, y: 6.22, w: W - 2 * M - 0.4, h: 0.68, fontFace: "Calibri", fontSize: 13, color: C.background1, valign: "middle", margin: 0, isTextBox: true });
rodape(s, [["A2", "Tendência"], ["A3", "Quebras"], ["A10", "Horário: testes"]]);
s.addNotes(`[~2 min] A demanda muda de patamar, não de tendência, e cada patamar coincide com uma mudança de negócio: horário, preço, público, nova máquina. Os números no gráfico marcam as quatro quebras estruturais de A3 (todas com p < 0,001). (1) O primeiro degrau (mai/2024) é a ampliação do horário: a base não registra o horário de funcionamento, então a janela foi inferida pela primeira e última venda do dia (10–19h → 07–23h); a demanda total subiu ${sinal(100 * (HR.irr_total - 1))} nas 8 semanas seguintes e o núcleo 10–19h não caiu (testes em A10; canibalização no slide 9). (2) e (3) vêm logo depois da redução de preços de julho e da alta de outubro: compatível com elasticidade-preço, mas não é um teste causal. (4) A nova clientela de 2025 leva o patamar a 14,4/dia; com ela, só 9% das vendas acontecem depois das 19h, então vale avaliar o custo de manter a máquina abastecida e ligada até 23h. A M2 abre em 08/02/2025 com ~6 vendas/dia e tem série própria. Os degraus explicam a série melhor que a reta (ΔBIC ${fmt(TD.dbic, 0)} na M1, ${fmt(TDT.dbic, 0)} no Total) e só ${TD.regimes_incl} de ${TD.regimes} regimes têm inclinação (A2). Com um único ano de dados, não dá para estimar sazonalidade anual; o que importa para a operação é saber em que patamar se está e detectar rápido quando ele muda.`);

// --- 7. padrão semanal (gráfico nativo)
s = conteudo("3. Mudanças estruturais", "3 · PADRÃO SEMANAL", "A M1 só passou a ter padrão semanal em 2025");
legenda(s, "Máquina 1 · vendas por dia da semana, % acima ou abaixo da média do período", M, 1.62, 8.2);
s.addChart(pres.charts.BAR, SEM.periodos.map((p, k) => ({ name: k === 0 ? "Antes de 30/01/2025" : "Desde 30/01/2025", labels: SEM.dias, values: SEM.valores[k].map(v => v / 100) })),
  eixos({ x: M, y: 1.95, w: 8.0, h: 4.95, barDir: "col", barGrouping: "clustered", barGapWidthPct: 55, chartColors: [COR.contexto, COR.m1],
          showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "+0%;-0%;0%", valAxisLabelFormatCode: "+0%;-0%;0%",
          valAxisMinVal: -0.6, valAxisMaxVal: 0.5, valAxisMajorUnit: 0.2, showLegend: true, legendPos: "t" }));
[[sinal(semDepois["fim de semana vs dias úteis (%)"]), `fim de semana × dias úteis desde 30/01/2025; antes, ${sinal(semAntes["fim de semana vs dias úteis (%)"])}`],
 [`R² ${fmt(semDepois["R² parcial do dia da semana"], 2)}`, `do dia da semana em 2025 (p < 0,001); antes, ${fmt(semAntes["R² parcial do dia da semana"], 2)} (p = ${fmt(semAntes.p, 2)})`],
 ["seg, qua, sex", `${fmt(SEM.valores[1][4], 0)}% a ${fmt(SEM.valores[1][0], 0)}% acima da média: agenda fixa do novo público`],
 [`${fmt(D.fds_m2_pct, 0)}%`, "das vendas da M2 caem no fim de semana: as duas máquinas se complementam"]]
  .forEach((c, i) => card(s, 8.95, 1.7 + i * 1.31, 3.78, 1.18, c[0], c[1], { tamanho: 22, rotulo: 12 }));
rodape(s, [["A2", "Teste de Fisher (período 7)"]]);
s.addNotes("[~1,5 min] O perfil desconta o nível de cada regime, para que um salto de patamar não pareça padrão semanal. Antes de 2025, nenhum regime da M1 tem efeito de dia da semana significativo. Com a nova clientela, o fim de semana passa a vender pouco mais da metade de um dia útil, e segunda, quarta e sexta se destacam, um padrão de agenda fixa (aulas, turnos ou expediente) que a base não permite confirmar. Consequências: abastecer a M1 antes de seg/qua/sex e reestimar o padrão semanal com janelas recentes, como faz o Holt-Winters de 8 semanas. O periodograma de Fisher, que confirma o período de 7 dias só dentro do regime de 2025, está no apêndice A2.");

// --- 8. ticket médio (gráfico nativo)
s = conteudo("3. Mudanças estruturais", "3 · TICKET MÉDIO", "O público mais recente gasta menos por compra");
legenda(s, "Ticket médio semanal (u.m. por venda)", M, 1.62, 8.2);
s.addChart(pres.charts.LINE, [
  { name: "Máquina 1", labels: D.ticket_semanal.rotulos, values: D.ticket_semanal.m1 },
  { name: "Máquina 2", labels: D.ticket_semanal.rotulos, values: D.ticket_semanal.m2 },
], eixos({ x: M, y: 1.95, w: 8.0, h: 4.95, chartColors: [COR.m1, COR.m2], lineSize: 2, lineDataSymbol: "circle", lineDataSymbolSize: 5,
           valAxisMinVal: 24, valAxisMaxVal: 36, valAxisMajorUnit: 2, valAxisLabelFormatCode: "0", catAxisLabelFrequency: 9,
           showLegend: true, legendPos: "t", displayBlanksAs: "gap" }));
[["2 tabelas", "em 2024, o ticket da M1 seguiu os preços: −13% a −21% em jul e +9% a +16% em out"],
 [sinal(TR.var_m1, 1), `ticket da M1 desde 30/01/2025 com a mesma tabela de preços (${fmt(TP["M1 · base anterior (mesma tabela)"].ticket)} → ${fmt(TP["M1 · desde a quebra (todos)"].ticket)}${UM})`],
 [`${fmt(D.mix_m1.Americano.antes, 0)}% → ${fmt(D.mix_m1.Americano.depois, 0)}%`, `das vendas da M1 são Americano (${fmt(D.mix_m1.Americano.preco, 0)}${UM}); o Latte cai de ${fmt(D.mix_m1.Latte.antes, 0)}% para ${fmt(D.mix_m1.Latte.depois, 0)}%`],
 [sinal(TR.m2_vs_m1), `ticket da M2 (${fmt(TP["M2 · desde a abertura"].ticket)}${UM}) frente à M1 no mesmo período: tabela até ${fmt(Math.abs(descMax), 0)}% mais barata`]]
  .forEach((c, i) => card(s, 8.95, 1.7 + i * 1.31, 3.78, 1.18, c[0], c[1], { tamanho: 22, rotulo: 12 }));
s.addNotes("[~1,5 min] A última mudança de tabela da M1 é de 06/10/2024, antes da quebra de 30/01/2025. Com os preços iguais nos dois períodos, toda a queda do ticket vem do mix: o novo público prefere Americano. A M2 vende os mesmos produtos mais barato (Americano with milk −19%, Cappuccino −16%) e tem 31% de vendas em dinheiro. Implicação: a receita cresce menos que o volume. Alavancas para testar: oferta com leite sobre o Americano e revisão da tabela da M2.");

// ================================================================== 9. canibalização + novos clientes: uma conclusão por análise
// (a versão detalhada, com IRR, IC, ρ e as coortes, está em A11)
pres.addSection({ title: "4. Canibalização" });
s = conteudo("4. Canibalização", "4 · CANIBALIZAÇÃO E NOVOS CLIENTES", "Crescimento por demanda nova; canibalização só parcial");
const analises = [
  { nome: "(a) Ampliação do horário", quando: "M1 · mai/2024", imagem: "canib_horario", num: sinal(100 * (HR.irr_total - 1)),
    rot: "de demanda total da M1; o núcleo 10–19h não caiu", concl: "Somou demanda; canibalização parcial não pode ser excluída" },
  { nome: "(b) Entrada da Máquina 2", quando: "fev/2025", imagem: "canib_m2", num: "14,2 → 14,4",
    rot: "vendas/dia da M1 antes × depois da entrada da M2", concl: "Sem perda líquida na M1; só um indício localizado à noite, confundido com a troca de público" },
  { nome: "(c) Salto de 2025", quando: "M1 · desde 30/01/2025", imagem: "clientes", num: "13,0 de 14,4",
    rot: "vendas/dia vêm de cartões novos (177 de 196 cartões nunca tinham comprado)", concl: "Demanda nova: uma nova clientela, de causa não identificável" },
];
const hB = 1.6, xR = 6.75, wR = W - M - xR;
analises.forEach((a, k) => {
  const y = 1.65 + k * (hB + 0.12);
  img(s, a.imagem, M, y + 0.04, xR - M - 0.2, hB - 0.08);
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: xR, y, w: wR, h: hB, rectRadius: 0.1, fill: { color: C.background2 }, line: { type: "none" }, objectName: `analise_${k}` });
  s.addText([{ text: a.nome, options: { bold: true, fontSize: 15, fontFace: "Cambria", color: C.text2 } }, { text: `   ${a.quando}`, options: { fontSize: 11, color: C.accent6 } }],
    { x: xR + 0.2, y: y + 0.1, w: wR - 0.4, h: 0.34, fontFace: "Calibri", valign: "middle", margin: 0, isTextBox: true });
  s.addText(a.num, { x: xR + 0.2, y: y + 0.48, w: 2.1, h: 0.46, fontFace: "Cambria", fontSize: 22, bold: true, color: C.accent1, valign: "middle", margin: 0, isTextBox: true, fit: "none" });
  s.addText(a.rot, { x: xR + 2.35, y: y + 0.46, w: wR - 2.55, h: 0.5, fontFace: "Calibri", fontSize: 12, color: C.text1, valign: "middle", margin: 0, isTextBox: true });
  s.addText(`→ ${a.concl}`, { x: xR + 0.2, y: y + 1.02, w: wR - 0.4, h: 0.5, fontFace: "Calibri", fontSize: 13, bold: true, color: C.text2, valign: "top", margin: 0, isTextBox: true });
});
rodape(s, [["A11", "Testes de canibalização"], ["A3", "Quebras"]]);
s.addNotes("[~2 min] Pergunta: a expansão gerou demanda nova ou só redistribuiu vendas? Três análises, em ordem cronológica, cada uma com a sua conclusão. O horário mudou em mai/2024 e a M2 só existe a partir de fev/2025, então a canibalização entre máquinas pela mudança de horário não é testável; por isso separamos (a) faixas horárias da M1 em 2024 e (b) efeito da entrada da M2 sobre a M1. (a) Com o horário ampliado, o total subiu 45% (p = 0,001) e o núcleo 10–19h ficou estável (IRR 0,92; p = 0,51); a incrementalidade de 0,84 (IC 0,34–1,23) não exclui canibalização parcial. (b) Abrir a M2 somou volume sem tirar vendas líquidas da M1 (IRR 1,06) e sem substituição diária (ρ = −0,11); há só um indício localizado à noite (~0,6/dia), confundido com a troca de público. (c) O proxy de cliente é o cartão (só a M1 tem): 177 de 196 cartões ativos depois de 30/01/2025 nunca tinham comprado, os cartões antigos caem de 7,0 para 1,4 vendas/dia e a nova coorte compra 3,8 vezes por cartão em 6 semanas, contra 1,5 das anteriores. É compatível com mudança de local ou de contexto, mas a causa não é identificável: cartões podem ser reemitidos. Todos os testes estão em A11.");

// ================================================================== 10–12. previsão
pres.addSection({ title: "5. Previsão" });
// --- 10. atributos e validação (requisito "desenvolvimento": atributos e decisões metodológicas no principal)
s = conteudo("5. Previsão", "5 · ATRIBUTOS E VALIDAÇÃO", "O teste nunca é visto: tudo é escolhido dentro do treino");
const tl = { x0: M + 1.9, x1: 12.6, y: 1.78 };
const tx = (d) => tl.x0 + (tl.x1 - tl.x0) * d;          // d em fração do período 01/03/2024–23/03/2025
const faixa = (rot, y, partes) => {
  s.addText(rot, { x: M, y, w: 1.8, h: 0.42, fontFace: "Calibri", fontSize: 13, bold: true, color: C.text2, valign: "middle", margin: 0, isTextBox: true });
  partes.forEach(p => s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: tx(p[0]), y: y + 0.05, w: tx(p[1]) - tx(p[0]) - 0.03, h: 0.32, rectRadius: 0.06, fill: { color: p[2] }, line: { type: "none" }, objectName: "faixa" }));
};
const f = (dias) => dias / 387;
faixa("Treino", tl.y, [[0, f(373), "CFC8C0"]]);
faixa("Validação interna", tl.y + 0.55, [[f(331), f(345), C.accent5], [f(345), f(359), C.accent5], [f(359), f(373), C.accent5]]);
faixa("Teste", tl.y + 1.1, [[f(374), f(388), C.accent1]]);
const bl = []; for (let k = 0; k < 11; k++) bl.push([f(234 + 14 * k), f(248 + 14 * k), k === 10 ? C.accent1 : C.accent6]);
faixa("Backtest", tl.y + 1.65, bl);
[["mar/24", 0], ["mai/24", f(61)], ["jul/24", f(122)], ["set/24", f(184)], ["nov/24", f(245)], ["jan/25", f(306)], ["mar/25", f(365)]].forEach(m =>
  s.addText(m[0], { x: tx(m[1]) - 0.4, y: tl.y + 2.12, w: 0.8, h: 0.3, align: "center", fontFace: "Calibri", fontSize: 11, color: C.accent6, margin: 0, isTextBox: true }));
const w4 = (W - 2 * M - 3 * 0.2) / 4;
caixaTexto(s, "Atributos, sem vazamento", [`${AT.n["Máquina 1"]} na M1: calendário, defasagens, médias móveis, quebras e comportamentais`, "só dados até a véspera; testes de vazamento no CI", "na RF, só as médias móveis importam (A6)"], M, 4.25, w4, 2.65, { tam: 13, tituloTam: 15 });
caixaTexto(s, "Dentro do treino", ["janela de estimação, transformação, ordens, exógenas e hiperparâmetros", "quebras e dummies redetectadas em cada origem"], M + (w4 + 0.2), 4.25, w4, 2.65, { tam: 13, tituloTam: 15 });
caixaTexto(s, "Critério de seleção", ["40% posto no teste + 35% backtest + 25% estabilidade", "filtro de negócio: |viés| ≤ 15%"], M + 2 * (w4 + 0.2), 4.25, w4, 2.65, { tam: 13, tituloTam: 15 });
caixaTexto(s, "Referência honesta", [`piso de ruído de Poisson: WAPE ~${fmt(Math.min(...Object.values(D.piso_ruido)), 0)}–${fmt(Math.max(...Object.values(D.piso_ruido)), 0)}%`, "ajuste abaixo do piso = sobreajuste"], M + 3 * (w4 + 0.2), 4.25, w4, 2.65, { tam: 13, tituloTam: 15 });
rodape(s, [["A4", "Modelos avaliados"], ["A5", "Atributos"], ["A8", "Métricas"]]);
s.addNotes(`[~2 min] Teste: últimos 14 dias. Validação: 3 dobras de 14 dias dentro do treino. Backtest: 11 blocos de 14 dias sem sobreposição, com tudo refeito em cada origem. Foram avaliados 2 baselines e 4 candidatos (apêndice A4, com os modelos testados e descartados). O teste entra na seleção final; por isso a estimativa honesta de desempenho é o backtest. Atributos (A5): ${AT.n["Máquina 1"]} na M1, ${AT.n["Total"]} no Total e ${AT.n["Máquina 2"]} na M2 (sem cartão), todos calculados só com dados até a véspera; cada modelo usa um subconjunto (o Holt-Winters não usa nenhum; o SARIMAX escolhe exógenas por AICc). Na Random Forest, a importância por permutação mostra que só as médias móveis importam (A6); nos modelos estatísticos, as dummies de quebra são essenciais (o AIC cai ${fmt(Math.abs(AT.daic_m1), 0)} na M1). Métricas completas no teste em A8.`);

// --- 11. modelos escolhidos: alternativas, justificativa e trade-off de cada escolha (o gráfico soma × modelo único está em A12)
s = conteudo("5. Previsão", "5 · MODELOS ESCOLHIDOS", "Qual previsão usar: um modelo por máquina; Total = soma");
s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: M, y: 1.6, w: W - 2 * M, h: 0.9, rectRadius: 0.1, fill: { color: C.background2 }, line: { type: "none" }, objectName: "alternativas" });
s.addText([{ text: "Alternativas, com a mesma validação em cada série: ", options: { bold: true, color: C.text2 } },
           { text: "2 baselines (naive sazonal, média 4 semanas) e 4 candidatos (Holt-Winters, SARIMAX, Poisson/NB2, Random Forest); no Total, modelo único × soma das máquinas.", options: { color: C.text1, breakLine: true } },
           { text: "Descartados antes, com evidência: Croston/SBA, GARCH, Prophet/TBATS/Theta, XGBoost/LightGBM, NHITS/NBEATSx (A4).", options: { color: C.accent6 } }],
  { x: M + 0.2, y: 1.63, w: W - 2 * M - 0.4, h: 0.84, fontFace: "Calibri", fontSize: 12.5, valign: "middle", margin: 0, isTextBox: true });
const iRF11 = EV.modelos.indexOf("Random Forest");
const escolhas = [
  { cor: COR.m1, nome: "Máquina 1", hist: "388 dias de histórico", modelo: "Holt-Winters (8 semanas)",
    num: `MASE ${fmt(m1d.mase_bt, 2)}`, numRot: `${bt_ganho_m1}% menos erro que o naive, em 11 janelas de 14 dias`,
    porque: "Reestima nível e semana com as últimas 8 semanas: acompanha o novo patamar e o padrão semanal de 2025.",
    tradeoff: `Não usa variáveis explicativas. A vantagem sobre o naive não é significativa após Holm (p = ${fmt(m1d.p_holm_naive, 2)}).`,
    erro: `±${fmt(m1d.mae)} vendas/dia (${fmt(m1d.wape, 0)}% do volume)` },
  { cor: COR.m2, nome: "Máquina 2", hist: "44 dias de histórico", modelo: "Random Forest regularizada",
    num: `${fmt(EV.m2_teste[iRF11], 2)}×`, numRot: `o erro do naive no teste (${fmt(EV.m2_val[iRF11], 2)}× na validação): o menor nas duas etapas`,
    porque: `Menor erro no teste: ${fmt(m2d.wape, 0)}% contra ${fmt(m2d.wape_media4, 0)}% da média de 4 semanas.`,
    tradeoff: `Só 30 dias de treino: sem backtest e sem diferença significativa para o naive (p ≥ ${fmt(D.dm_m2_p_min, 2)}). Ler como nível médio, com faixa larga.`,
    erro: `±${fmt(m2d.mae)} vendas/dia (${fmt(m2d.wape, 0)}%); piso de ruído ≈ ${fmt(D.piso_ruido["Máquina 2"], 0)}%` },
  { cor: COR.total, nome: "Total", hist: "soma das previsões", modelo: "Soma M1 + M2",
    num: `−${ganhoSoma("demanda")}% / −${ganhoSoma("receita")}%`, numRot: "erro médio da soma × modelo único do Total, na demanda / na receita (teste)",
    porque: "A operação decide por máquina, e o modelo único achata padrões opostos: M1 em dias úteis, M2 no fim de semana.",
    tradeoff: `1 janela de 14 dias e RMSE maior na demanda (${fmt(sd.RMSE, 1)} × ${fmt(sx.RMSE, 1)}). O modelo único (SARIMAX) segue em paralelo como alarme.`,
    erro: `±${fmt(TS("demanda", "Soma M1 + M2").MAE)} vendas/dia (${fmt(TS("demanda", "Soma M1 + M2")["WAPE_%"], 0)}% do volume)` },
];
escolhas.forEach((c, i) => {
  const x = M + i * 4.1, y = 2.62, w = 3.85, h = 4.28, xi = x + 0.22, wi = w - 0.44;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.12, fill: { color: C.background2 }, line: { type: "none" }, objectName: `escolha_${i}` });
  circulo(s, xi, y + 0.14, 0.36, c.cor);
  s.addText(c.nome, { x: xi + 0.48, y: y + 0.09, w: 1.6, h: 0.46, fontFace: "Cambria", fontSize: 17, bold: true, color: C.text2, valign: "middle", margin: 0, isTextBox: true });
  s.addText(c.hist, { x: x + w - 1.95, y: y + 0.09, w: 1.73, h: 0.46, fontFace: "Calibri", fontSize: 11, color: C.accent6, align: "right", valign: "middle", margin: 0, isTextBox: true });
  s.addText(c.modelo, { x: xi, y: y + 0.53, w: wi, h: 0.36, fontFace: "Cambria", fontSize: 16, bold: true, color: C.accent1, valign: "middle", margin: 0, isTextBox: true });
  s.addText(c.num, { x: xi, y: y + 0.9, w: wi, h: 0.42, fontFace: "Cambria", fontSize: 22, bold: true, color: C.text2, valign: "middle", margin: 0, isTextBox: true, fit: "none" });
  s.addText(c.numRot, { x: xi, y: y + 1.32, w: wi, h: 0.42, fontFace: "Calibri", fontSize: 11.5, color: C.accent6, valign: "top", margin: 0, isTextBox: true });
  s.addText([{ text: "Por que", options: { bold: true, fontSize: 11, color: C.accent1, breakLine: true } }, { text: c.porque, options: { fontSize: 12.5, color: C.text1 } }],
    { x: xi, y: y + 1.78, w: wi, h: 0.86, fontFace: "Calibri", valign: "top", margin: 0, isTextBox: true });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: x + 0.12, y: y + 2.66, w: w - 0.24, h: 0.98, rectRadius: 0.08, fill: { color: C.background1 }, line: { type: "none" }, objectName: `tradeoff_${i}` });
  s.addText([{ text: "Trade-off e ressalva", options: { bold: true, fontSize: 11, color: C.accent1, breakLine: true } }, { text: c.tradeoff, options: { fontSize: 11.5, color: C.text1 } }],
    { x: xi, y: y + 2.71, w: wi, h: 0.9, fontFace: "Calibri", valign: "top", margin: 0, isTextBox: true });
  s.addText([{ text: "Erro típico no teste", options: { fontSize: 11, color: C.accent6, breakLine: true } }, { text: c.erro, options: { fontSize: 12, bold: true, color: C.text2 } }],
    { x: xi, y: y + 3.7, w: wi, h: 0.5, fontFace: "Calibri", valign: "middle", margin: 0, isTextBox: true });
});
rodape(s, [["A4", "Modelos"], ["A7", "Evidência"], ["A12", "Soma × único"], ["A8", "Métricas"]]);
s.addNotes(`[~2,5 min] Resposta direta à pergunta 'qual modelo e se a previsão deve ser conjunta'. Em cada série, os mesmos 2 baselines e 4 candidatos passaram pela mesma validação; a escolha combina erro no teste, posto no backtest, estabilidade e o filtro de viés. Máquina 1: Holt-Winters de 8 semanas, o melhor em 11 janelas (MASE ${fmt(m1d.mase_bt, 2)}); o trade-off é não usar variáveis explicativas e a vantagem sobre o naive não ser significativa após Holm (p = ${fmt(m1d.p_holm_naive, 2)}). Máquina 2: Random Forest regularizada, o menor erro na validação interna e no teste; com só 30 dias de treino não há backtest, e nenhum modelo difere do naive com significância (Diebold-Mariano p ≥ ${fmt(D.dm_m2_p_min, 2)}), por isso a previsão da M2 é lida como nível médio com faixa larga e não tem a mesma confiança da M1. Total: a soma das máquinas, não um modelo único: a operação decide por máquina, o total fica sempre igual à soma das partes, e a soma errou menos (${ganhoSoma("demanda")}% na demanda e ${ganhoSoma("receita")}% na receita). Ressalvas da soma: é 1 janela de 14 dias, sem backtest possível (a M2 só existe desde 08/02/2025); na demanda, a soma tem MAE e WAPE menores, mas RMSE maior e R² menor (${fmt(sd.R2, 2)} × ${fmt(sx.R2, 2)}) que o SARIMAX. O modelo único do Total (SARIMAX) segue em paralelo como alarme de coerência e plano B: prevê ${fmt(ft.demanda.media)} vendas/dia, e divergência grande da soma = investigar. Reavaliar com backtest quando a M2 tiver 3 meses e depois testar reconciliação hierárquica (MinT). Gráfico e números da comparação em A12; evidência por máquina em A7; métricas completas em A8.`);

// --- 12. previsões
s = conteudo("5. Previsão", "5 · PREVISÕES", `Próximos 14 dias: ~${fmt(fs_.demanda.media, 0)} vendas e ~${mil(fs_.receita.media)} u.m. por dia`);
img(s, "prev_m1_m2", M, 1.6, 8.15, 4.3, "esquerda");
s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: M, y: 6.02, w: 8.15, h: 0.88, rectRadius: 0.1, fill: { color: C.background2 }, line: { type: "none" }, objectName: "como_ler" });
s.addText([{ text: "Como ler: ", options: { bold: true, color: C.text2 } },
           { text: "acima, Máquina 1 (Holt-Winters); abaixo, Máquina 2 (Random Forest); faixas = intervalos de 80% e 95%.", options: { color: C.text1, breakLine: true } },
           { text: "Incerteza: ", options: { bold: true, color: C.text2 } },
           { text: `no teste, o intervalo de 80% cobriu ${fmt(q1["cobertura_80_%"], 0)}% dos dias na M1 e ${fmt(q2["cobertura_80_%"], 0)}% na M2; o de 95%, todos (calibrados ou um pouco largos).`, options: { color: C.text1 } }],
  { x: M + 0.18, y: 6.06, w: 8.15 - 0.36, h: 0.8, fontFace: "Calibri", fontSize: 12, valign: "middle", margin: 0, isTextBox: true });
const xP = 9.0, wP = W - M - xP;
s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: xP, y: 1.7, w: wP, h: 4.12, rectRadius: 0.1, fill: { color: C.background2 }, line: { type: "none" }, objectName: "caixa_prev" });
const linhasPrev = [
  ["Máquina 1 · Holt-Winters", f1, `${fmt(f1.demanda.media)} vendas/dia · erro típico ±${fmt(m1d.mae)} (${fmt(m1d.wape, 0)}%)`],
  ["Máquina 2 · Random Forest", f2, `${fmt(f2.demanda.media)} vendas/dia · erro típico ±${fmt(m2d.mae)} (${fmt(m2d.wape, 0)}%)`, "só 44 dias de histórico: ler como nível médio, com faixa larga"],
  ["Total (soma M1 + M2)", fs_, `${fmt(fs_.demanda.media)} vendas/dia · erro típico ±${fmt(TS("demanda", "Soma M1 + M2").MAE)} (${fmt(TS("demanda", "Soma M1 + M2")["WAPE_%"], 0)}%)`],
  ["Modelo único do Total (checagem)", ft, `${fmt(ft.demanda.media)} vendas/dia`],
];
s.addText([{ text: "Próximos 14 dias", options: { bold: true, fontSize: 16, color: C.text2, breakLine: true } },
           { text: periodoFut, options: { fontSize: 12, color: C.accent6, breakLine: true, paraSpaceAfter: 8 } },
           ...linhasPrev.flatMap(([rot, v, extra, nota], k) => [
             { text: rot, options: { bold: true, fontSize: 13, color: C.text2, breakLine: true } },
             { text: `${mil(v.demanda.soma)} vendas · ${mil(v.receita.soma)}${UM}`, options: { fontSize: 14, color: C.text1, breakLine: true } },
             { text: extra, options: { fontSize: 12, color: C.accent6, breakLine: nota ? true : k < linhasPrev.length - 1, paraSpaceAfter: nota ? 0 : 8 } },
             ...(nota ? [{ text: nota, options: { fontSize: 12, italic: true, color: C.accent1, breakLine: k < linhasPrev.length - 1, paraSpaceAfter: 8 } }] : [])])],
  { x: xP + 0.15, y: 1.82, w: wP - 0.3, h: 3.9, fontFace: "Calibri", valign: "top", margin: 2, isTextBox: true });
s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: xP, y: 5.95, w: wP, h: 0.95, rectRadius: 0.1, fill: { color: C.text2 }, line: { type: "none" }, objectName: "uso_prev" });
s.addText([{ text: "Reposição: ", options: { bold: true, color: C.background1 } }, { text: "limite superior da faixa de 80% por máquina. ", options: { color: C.background1 } },
           { text: "Caixa: ", options: { bold: true, color: C.background1 } }, { text: "a soma central.", options: { color: C.background1 } }],
  { x: xP + 0.18, y: 5.98, w: wP - 0.36, h: 0.89, fontFace: "Calibri", fontSize: 13, valign: "middle", margin: 0, isTextBox: true });
rodape(s, [["A8", "Métricas e cobertura dos intervalos"]]);
s.addNotes("[~2 min] A Máquina 1 repete o padrão semanal de 2025 (vales no fim de semana); a Máquina 2 fica quase plana, com faixa larga, porque só tem 44 dias de histórico: a previsão dela não tem a mesma confiança da M1 e deve ser lida como nível médio. Previsões, intervalos e métricas de cada série ficam em reports/forecasts (machine_1, machine_2, total). Para reposição, usar o limite superior da faixa de 80% por máquina; para caixa, a soma central. A diferença entre a soma e o modelo único na receita (~53 u.m./dia) é o tipo de divergência que o monitoramento acompanha. Cobertura dos intervalos de cada modelo no apêndice A8.");

// ================================================================== 13–15. decisões de negócio
pres.addSection({ title: "6. Decisões de negócio" });
s = conteudo("6. Decisões de negócio", "6 · RECOMENDAÇÕES E DECISÕES", "Seis decisões sobre produtos, clientes e operação");
const conc = [
  ["Enxugar a cauda da M2", `${CM2["produtos com ≤ 5 vendas"]} dos ${CM2["produtos"]} produtos venderam até 5 vezes em 44 dias e somam ${fmt(CM2["% receita da cauda"], 0)}% da receita. Menos itens = menos insumo parado ou vencido e abastecimento mais simples.`],
  ["Cortar por insumo", `Um insumo só sai de linha se todos os seus produtos saírem: baunilha (${FAM("Máquina 2", "baunilha").produtos} itens, ${FAM("Máquina 2", "baunilha").vendas} vendas em 44 dias) é a primeira candidata. Whiskey fica: ${fmt(FAM("Máquina 2", "whiskey (aroma)").pct_receita, 0)}% da receita da M2.`],
  ["Nunca faltar o essencial", `Leite está em ${fmt(FAM("Máquina 1", "leite").pct_vendas, 0)}% das vendas da M1 e ${fmt(FAM("Máquina 2", "leite").pct_vendas, 0)}% da M2; ${CTOT["produtos para 80% da receita"]} produtos fazem 80% da receita. Estoque de segurança pelo limite superior da previsão.`],
  ["Recuperar o ticket", `O público de 2025 troca Latte por Americano (${fmt(D.mix_m1.Americano.antes, 0)}% → ${fmt(D.mix_m1.Americano.depois, 0)}% das vendas) e gasta ${fmt(Math.abs(TR.var_m1))}% menos. Testar oferta com leite e combos sobre o Americano.`],
  ["Rever a tabela da M2", `Mesmos produtos até ${fmt(Math.abs(descMax), 0)}% mais baratos que na M1 e ticket ${fmt(Math.abs(TR.m2_vs_m1), 0)}% menor. Alinhar preços só com teste: a elasticidade não é estimável com estes dados.`],
  ["Operar por dia e horário", `M1: seg/qua/sex fortes, fim de semana ${sinal(semDepois["fim de semana vs dias úteis (%)"])} e noite de ${fmt(TP["M1 · base anterior (mesma tabela)"].noite, 0)}% para ${fmt(TP["M1 · desde a quebra (todos)"].noite, 0)}% das vendas. M2: ${fmt(D.fds_m2_pct, 0)}% no fim de semana. Abastecer a M1 no início da semana e a M2 na sexta.`],
];
conc.forEach((c, i) => {
  const x = M + (i % 3) * 4.1, y = 1.7 + Math.floor(i / 3) * 2.68, w = 3.85, h = 2.5;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.12, fill: { color: C.background2 }, line: { type: "none" }, objectName: `conc_${i}` });
  circulo(s, x + 0.22, y + 0.2, 0.5, C.accent1, String(i + 1));
  s.addText(c[0], { x: x + 0.85, y: y + 0.17, w: w - 1.0, h: 0.56, fontFace: "Cambria", fontSize: 16, bold: true, color: C.text2, valign: "middle", margin: 0, isTextBox: true });
  s.addText(c[1], { x: x + 0.22, y: y + 0.85, w: w - 0.44, h: h - 0.97, fontFace: "Calibri", fontSize: 14, color: C.text1, valign: "top", margin: 0, isTextBox: true });
});
s.addNotes("[~2 min] As seis conclusões saem dos mesmos dados da previsão. As 1 a 3 reduzem custo (insumo parado, vencimento, ruptura do essencial); as 4 e 5 atacam receita (ticket); a 6 é operação. Nenhuma tem valor financeiro estimado, porque faltam custo e margem; por isso viram pilotos medidos contra a previsão.");

// ================================================================== 14–16. produção, limitações e encerramento
pres.addSection({ title: "7. Produção e próximos passos" });
// --- 14. arquitetura para produção (resumo de A9; distingue o que já está implementado do que é proposta)
s = conteudo("7. Produção e próximos passos", "7 · ARQUITETURA PARA PRODUÇÃO", "Lote diário: o núcleo já roda, a operação é proposta");
legenda(s, "Preenchido = implementado no repositório (python -m src.pipeline) · tracejado = proposta de implantação (docs/architecture.md)", M, 1.6, W - 2 * M);
const fluxo = [
  ["Logs de vendas", "CSV versionado com SHA-256", true],
  ["Validação e contrato", "colunas, tipos e domínios (validar_bruto)", true],
  ["Séries diárias", "calendário contínuo; Total = M1 + M2", true],
  ["Quebras + atributos", "Binseg + BIC só até a origem; testes de vazamento", true],
  ["Treino e seleção", "6 modelos; backtest de 11 origens; escore de seleção", true],
  ["Registro e versão", "joblib + metadados com hash de dados e configuração", true],
  ["Previsão de 14 dias", "intervalos de 80% e 95%", true],
  ["Monitoramento, drift e retreino", "alertas e gatilho de retreino (volta à 5)", false],
];
const wF = 1.38, gF = (W - 2 * M - 8 * wF) / 7;
fluxo.forEach(([nome, sub, impl], i) => {
  const x = M + i * (wF + gF), y = 1.95, h = 0.85;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: wF, h, rectRadius: 0.1, fill: { color: impl ? C.text2 : C.background1 },
    line: impl ? { type: "none" } : { color: HEX.accent1, width: 1.5, dashType: "dash" }, objectName: `fluxo_${i + 1}` });
  s.addText([{ text: String(i + 1), options: { fontSize: 11, bold: true, color: impl ? "E3C3A8" : C.accent1, breakLine: true } },
             { text: nome, options: { fontSize: 12, bold: true, color: impl ? C.background1 : C.accent1 } }],
    { x: x + 0.06, y, w: wF - 0.12, h, align: "center", valign: "middle", fontFace: "Calibri", margin: 0, isTextBox: true });
  s.addText(sub, { x, y: y + h + 0.05, w: wF, h: 0.55, align: "center", valign: "top", fontFace: "Calibri", fontSize: 10.5, color: C.accent6, margin: 0, isTextBox: true });
  if (i < fluxo.length - 1) s.addShape(pres.shapes.LINE, { x: x + wF + 0.02, y: y + h / 2, w: gF - 0.04, h: 0, line: { color: HEX.accent6, width: 1.5, endArrowType: "triangle" }, objectName: `seta_fluxo_${i + 1}` });
});
const pilares = [
  ["Observabilidade", "proposta", ["WAPE, MAE e viés por máquina, em janelas de 14 e 28 dias", "alerta: WAPE de 28 dias > 1,25 × o do backtest ou |viés| > 15%", "cobertura do intervalo de 80%; soma × modelo único"]],
  ["Drift: dados e público", "proposta", ["dados: PSI/KS de demanda, ticket e horário; χ² do mix", "comportamento: quebra online + CUSUM dos resíduos", "esquema: coluna nova ou ausente bloqueia a carga"]],
  ["Critérios de retreino", "proposta", ["semanal; seleção completa com backtest todo mês", "antecipado: alerta por 7 dias seguidos ou nova quebra", "nova máquina: modelo plano até ~8 semanas"]],
  ["Versão, rastreio e promoção", "parte implementada", ["feito: metadados com hash de dados e configuração; CI a cada push", "proposta: previsões gravadas com a versão do modelo e dos dados", "shadow mode de 1–2 semanas antes de promover"]],
];
const wPl = (W - 2 * M - 3 * 0.2) / 4;
pilares.forEach(([tit, status, itens], i) => {
  const x = M + i * (wPl + 0.2), y = 3.6, h = 2.4;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w: wPl, h, rectRadius: 0.1, fill: { color: C.background2 }, line: { type: "none" }, objectName: `pilar_${i}` });
  s.addText([{ text: tit, options: { bold: true, fontSize: 14, color: C.text2, breakLine: true } },
             { text: status, options: { fontSize: 10.5, italic: true, color: C.accent1, breakLine: true, paraSpaceAfter: 4 } },
             ...itens.map((t, k) => ({ text: t, options: { bullet: true, fontSize: 11.5, color: C.text1, breakLine: k < itens.length - 1, paraSpaceAfter: 3 } }))],
    { x: x + 0.15, y: y + 0.1, w: wPl - 0.3, h: h - 0.2, fontFace: "Calibri", valign: "top", margin: 2, isTextBox: true });
});
s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: M, y: 6.12, w: W - 2 * M, h: 0.78, rectRadius: 0.1, fill: { color: C.text2 }, line: { type: "none" }, objectName: "escala" });
s.addText([{ text: "Escala: ", options: { bold: true } },
           { text: "batch, não streaming (dezenas de vendas por dia, decisões diárias); o pipeline completo roda em ~10 min numa única máquina, e as séries independentes paralelizam para novas máquinas. " },
           { text: "Manutenção: ", options: { bold: true } },
           { text: "parâmetros em config.yaml, logs em arquivo e testes de invariantes (Total = M1 + M2, sem vazamento)." }],
  { x: M + 0.2, y: 6.12, w: W - 2 * M - 0.4, h: 0.78, fontFace: "Calibri", fontSize: 12, color: C.background1, valign: "middle", margin: 0, isTextBox: true });
rodape(s, [["A9", "Arquitetura detalhada"]]);
s.addNotes("[~2 min] Resumo do apêndice A9 e de docs/architecture.md. O que já está implementado no repositório: o pipeline encadeado (python -m src.pipeline), da carga com validação de contrato e hash dos arquivos brutos até a previsão de 14 dias com intervalos, passando por séries diárias, quebras, atributos com testes de vazamento, treino com validação interna e backtest de 11 origens, e o registro dos modelos em joblib com metadados (hash dos dados e da configuração, janela de treino, especificação); os testes rodam no CI a cada push. O que é proposta de implantação: agendamento diário, monitoramento com alertas, detecção de drift, retreino por calendário e por gatilho, previsões gravadas com versão do modelo e dos dados, e shadow mode antes de promover um modelo. Batch, e não streaming, porque são dezenas de vendas por dia e as decisões são diárias. Para muitas máquinas, vale testar um modelo global; com só 2 máquinas e uma delas com 30 dias, o desenho conjunto piorou a M2 (modelagens_testes, seção 8.3). Os nomes de ferramentas de mercado ficam como exemplos em docs/architecture.md; o desenho não depende deles.");

// --- 15. limitações em três grupos + próximos passos com a forma de medir cada um
s = conteudo("7. Produção e próximos passos", "7 · LIMITAÇÕES E PRÓXIMOS PASSOS", "Limitações e próximos passos: o que falta e como avançar");
const grupos = [
  ["Dados de negócio ausentes", ["Sem custo e margem por produto: o corte da cauda é por giro, não por lucro", "Sem estoque e rupturas: venda baixa pode ser falta do produto na máquina", "Local e país não informados (feriados genéricos); M2 sem cartão"]],
  ["Cliente e preço", ["Substituição desconhecida: sem um item, o cliente compra outro ou desiste?", "Preço nunca testado com grupo de controle: elasticidade desconhecida", "M2 com só 44 dias: a cauda pode ser efeito novidade do cardápio"]],
  ["Previsão e riscos", ["1 ano de histórico: sem sazonalidade anual", "Quebras detectadas com 3–4 semanas de atraso: risco de prever no patamar antigo", "M2 e Total sem backtest: a M2 é lida como nível médio, com faixa larga"]],
];
grupos.forEach(([tit, itens], i) => caixaTexto(s, tit, itens, M, 1.7 + i * 1.75, 5.95, 1.63, { tam: 12.5, tituloTam: 15, espaco: 3 }));
const fases2 = [
  ["Próximas 4 semanas", `previsão por máquina em paralelo (shadow mode) · cadastrar feriados e fechamentos · levantar custo e margem dos ${CTOT["produtos"]} produtos`, "WAPE e viés da previsão contra o realizado, por máquina"],
  ["1 a 3 meses", "piloto na M2 sem a família baunilha e os itens com até 3 vendas · oferta com leite sobre o Americano · abastecimento por dia da semana", "vendas e receita realizadas × previstas sem a mudança"],
  ["3 meses ou mais", "teste controlado da tabela da M2 · backtest da M2 e do Total · previsão dos 5 produtos principais para comprar insumos", "grupo de controle no teste de preço; MASE em janelas, como na M1"],
];
fases2.forEach((fz, i) => {
  const x = M + 6.18, y = 1.7 + i * 1.47, w = W - M - x, h = 1.37;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.1, fill: { color: C.background2 }, line: { type: "none" }, objectName: `fase_${i}` });
  circulo(s, x + 0.2, y + 0.2, 0.45, C.accent1, String(i + 1));
  s.addText([{ text: fz[0], options: { bold: true, fontSize: 15, color: C.text2, fontFace: "Cambria", breakLine: true } },
             { text: fz[1], options: { fontSize: 12.5, color: C.text1, breakLine: true, paraSpaceAfter: 3 } },
             { text: "Medida: ", options: { bold: true, fontSize: 12, color: C.accent1 } },
             { text: fz[2], options: { fontSize: 12, color: C.accent1 } }],
    { x: x + 0.8, y: y + 0.1, w: w - 0.95, h: h - 0.18, fontFace: "Calibri", valign: "top", margin: 0, isTextBox: true });
});
s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: M + 6.18, y: 6.13, w: W - M - (M + 6.18), h: 0.77, rectRadius: 0.1, fill: { color: C.text2 }, line: { type: "none" }, objectName: "mensagem" });
s.addText("Cada piloto é medido contra a previsão por máquina: o que se esperava vender sem a mudança.",
  { x: M + 6.38, y: 6.13, w: W - M - (M + 6.18) - 0.4, h: 0.77, fontFace: "Calibri", fontSize: 14, bold: true, color: C.background1, valign: "middle", margin: 0, isTextBox: true });
rodape(s, [["A1", "Qualidade dos dados"], ["A9", "Monitoramento e retreino"]]);
s.addNotes("[~1,5 min] À esquerda, as limitações em três grupos: dados de negócio que faltam (a principal lacuna é custo e margem por produto), comportamento do cliente e preço (substituição e elasticidade desconhecidas) e riscos da previsão (histórico curto, atraso na detecção de quebras, M2 e Total sem backtest). À direita, a sequência: primeiro operar a previsão em paralelo e coletar o que falta; depois pilotos de baixo risco (cauda da M2, oferta sobre o Americano); por fim, preço com teste controlado. Cada horizonte tem a sua medida; medir cada piloto contra a previsão evita confundir o efeito da ação com a variação normal da demanda. Nenhuma recomendação tem ganho financeiro comprovado, porque faltam custo e margem. Detalhes técnicos das limitações em docs/limitations.md; monitoramento e retreino em A9.");

// --- 16. encerramento com os aprendizados
s = pres.addSlide({ masterName: "CAPA", sectionTitle: "7. Produção e próximos passos" });
s.addText("Prever por máquina, enxugar o cardápio e proteger o essencial", { placeholder: "title" });
s.addText([{ text: "O que aprendemos", options: { bold: true, fontSize: 17, fontFace: "Cambria", color: C.background1, breakLine: true, paraSpaceAfter: 6 } },
           { text: "A demanda muda de patamar com o negócio (horário, preço, público): planejar pelo nível atual e vigiar quebras", options: { bullet: { type: "number" }, fontSize: 15, color: C.background2, breakLine: true, paraSpaceAfter: 4 } },
           { text: "Públicos diferentes pedem um modelo por máquina; o Total sai da soma das duas", options: { bullet: { type: "number" }, fontSize: 15, color: C.background2, breakLine: true, paraSpaceAfter: 4 } },
           { text: "Com pouco histórico (M2), o honesto é prever o nível com faixa larga e reavaliar com mais dados", options: { bullet: { type: "number" }, fontSize: 15, color: C.background2 } }],
  { x: M, y: 4.3, w: 11.5, h: 1.75, fontFace: "Calibri", valign: "top", margin: 0, isTextBox: true });
s.addText("Código, notebooks, testes, relatórios e documentação em case_coffee_sales/ · python -m src.pipeline reproduz todos os resultados",
  { x: M, y: 6.4, w: 11.5, h: 0.4, fontFace: "Calibri", fontSize: 12, color: C.background2, valign: "middle", margin: 0, isTextBox: true });
s.addNotes("[~0,5 min] Fechamento com os três aprendizados, todos ligados a resultados mostrados: os degraus (slide 6), a soma por máquina (slide 11) e a cautela com a M2 (slides 11 e 12). Abrir para perguntas. O índice do apêndice (próximo slide) tem links para cada detalhe técnico; em cada slide do apêndice, o rodapé volta para o índice.");

// ================================================================== apêndice técnico
pres.addSection({ title: "Apêndice técnico" });
// --- índice (com links): A1–A9 = base técnica; A10–A12 = detalhe que saiu dos slides 6, 9 e 11
s = conteudo("Apêndice técnico", "APÊNDICE TÉCNICO", "Apêndice: o detalhe técnico para as perguntas");
legenda(s, "A1–A9: base técnica · A10–A12: testes e comparações que saíram dos slides 6, 9 e 11 · clique em um cartão para abrir o slide; o rodapé de cada slide volta para cá", M, 1.6, W - 2 * M);
const indice = [
  ["A1", "Qualidade dos dados", "ausentes, zeros, outliers e duplicados: nada removido, cada decisão justificada", "apoia o slide 3"],
  ["A2", "Tendência e sazonalidade", "degraus × reta (ΔBIC), periodograma de Fisher, ciclo e inércia", "apoia os slides 6 e 7"],
  ["A3", "Quebras estruturais", "quatro mudanças de nível, todas com p < 0,001 (Sup-Wald de Andrews)", "apoia os slides 6 e 9"],
  ["A4", "Modelos avaliados", "2 baselines, 4 candidatos e os modelos testados e descartados", "apoia os slides 10 e 11"],
  ["A5", "Engenharia de atributos", `${AT.n["Máquina 1"]} atributos na M1 em 6 grupos, todos só com dados até a véspera`, "apoia o slide 10"],
  ["A6", "Importância das variáveis", "permutação fora da amostra: só as médias móveis importam", "apoia o slide 10"],
  ["A7", "Evidência por máquina", "MASE em 11 janelas (M1); validação e teste contra o naive (M2)", "apoia o slide 11"],
  ["A8", "Métricas de qualidade", "MAE, RMSE, MAPE, WAPE, R², viés e cobertura dos intervalos no teste", "apoia os slides 11 e 12"],
  ["A9", "Arquitetura para produção", "pipeline, monitoramento, drift, retreino e versionamento", "apoia o slide 14"],
  ["A10", "Ampliação do horário", "janela inferida, data detectada e efeito em ±8 semanas (IRR, p)", "detalhe do slide 6"],
  ["A11", "Testes de canibalização", "IRR, incrementalidade, correlação e coortes de cartões", "detalhe do slide 9"],
  ["A12", "Soma × modelo único", "gráfico do teste, erro na demanda e na receita e ressalvas", "detalhe do slide 11"],
];
indice.forEach(([sig, nome, desc, apoio], i) => {
  const x = M + (i % 3) * 4.1, y = 1.98 + Math.floor(i / 3) * 1.22, w = 3.85, h = 1.1, detalhe = i >= 9;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.1, fill: { color: C.background2 }, line: { type: "none" }, objectName: `indice_${sig}` });
  marcador(s, x + 0.17, y + (h - 0.54) / 2, 0.54, detalhe ? C.accent5 : C.accent1, C.background1, sig, sig.length > 2 ? 13 : 15);
  s.addText([{ text: nome, options: { bold: true, fontSize: 14, color: C.text2, fontFace: "Cambria", breakLine: true, paraSpaceAfter: 1 } },
             { text: desc, options: { fontSize: 11.5, color: C.text1, breakLine: true, paraSpaceAfter: 1 } },
             { text: apoio, options: { fontSize: 10.5, color: C.accent6 } }],
    { x: x + 0.86, y: y + 0.06, w: w - 0.98, h: h - 0.12, fontFace: "Calibri", valign: "middle", margin: 0, isTextBox: true });
  link(s, x, y, w, h, N[sig], `Ir para ${sig}: ${nome}`);
});
s.addNotes("Índice do apêndice, para navegar durante as perguntas. A1–A9 formam a base técnica; A10–A12 guardam o detalhe estatístico que saiu dos slides 6, 9 e 11 para deixá-los mais leves. Cada cartão leva ao slide correspondente (no modo apresentação); o rodapé de cada slide do apêndice volta para cá. Os slides principais também têm, no rodapé, links para o apêndice que os detalha.");

// --- A1. qualidade dos dados (v1: slide 11)
s = apendice("A1", "QUALIDADE DOS DADOS", "Ausentes, zeros, outliers e duplicados: nada removido");
const qual = [
  ["Ausentes", "89 = 100%", "dos card nulos são vendas em dinheiro; a M2 não tem a coluna card nem existe antes de 08/02/2025", "Decisão: nada imputado; clientes só na M1"],
  ["Zeros", "7 dias", "na M1, implausíveis como acaso (P < 0,2%): provável fechamento. 5 dos 7 caem em feriado ou fim de semana", "Decisão: mantidos; viram alerta de monitoramento"],
  ["Outliers", `${AQ.k} de ${AQ.dias}`, `quedas em feriado ou fim de semana (${fmt(AQ.pct, 0)}% × ${fmt(AQ.esperado, 0)}% esperado; p = ${fmt(AQ.p, 2)}). Os ${AP.dias} picos são dias úteis de fev/2025`, "Decisão: nada removido; calendário de feriados em produção"],
  ["Duplicados", `${DU.exatas_m2} na M2`, `no mesmo segundo, como em outros pedidos com vários itens. Na M1, as recompras do mesmo item levam ~${fmt(DU.med_mesmo, 0)} s (nenhuma < 30 s), como as de itens diferentes`, "Decisão: mantidos; parecem 2 unidades na mesma compra, sem ID de pedido para confirmar"],
];
const wq = (W - 2 * M - 3 * 0.25) / 4;
qual.forEach((q, i) => {
  const x = M + i * (wq + 0.25), xi = x + 0.22, wi = wq - 0.44;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 1.75, w: wq, h: 5.1, rectRadius: 0.12, fill: { color: C.background2 }, line: { type: "none" }, objectName: `q_${i}` });
  s.addText(q[0], { x: xi, y: 1.9, w: wi, h: 0.45, fontFace: "Cambria", fontSize: 20, bold: true, color: C.text2, margin: 0, isTextBox: true });
  s.addText(q[1], { x: xi, y: 2.4, w: wi, h: 0.7, fontFace: "Cambria", fontSize: 28, bold: true, color: C.accent1, margin: 0, isTextBox: true, fit: "none" });
  s.addText(q[2], { x: xi, y: 3.2, w: wi, h: 2.35, fontFace: "Calibri", fontSize: 14, color: C.text1, valign: "top", margin: 0, isTextBox: true });
  s.addText(q[3], { x: xi, y: 5.6, w: wi, h: 1.1, fontFace: "Calibri", fontSize: 14, bold: true, color: C.text2, valign: "top", margin: 0, isTextBox: true });
});
s.addNotes(`Quedas = 7 dias sem venda da M1 + 26/12/2024. Feriados: 01/05/2024 e 01/01/2025; fins de semana: 04–05/05/2024 e 19/01/2025; 26/12 é o dia seguinte ao Natal. A comparação com o calendário importa: 2 de cada 7 dias já são fim de semana. Os picos, ao contrário, são dias úteis e acompanham a chegada do novo público. Duplicados: as ${DU.exatas_m2} duplicatas exatas da M2 (dois Americano with milk em dinheiro, no mesmo segundo) estão entre ${DU.grupos_m2} grupos de registros no mesmo segundo, ${DU.grupos_dif_m2} deles com produtos diferentes: padrão de pedido com vários itens. Na M1 não há duplicata exata; há ${DU.recompras_m1} recompras do mesmo produto pelo mesmo cartão em até 5 min (${fmt(DU.pct_m1, 1)}% das vendas), com intervalo mínimo de ${fmt(DU.min_mesmo, 0)} s e mediano de ${fmt(DU.med_mesmo, 0)} s, parecido com as recompras de produto diferente (${fmt(DU.med_dif, 0)} s). Um registro em dobro ficaria a ~0 s. Não dá para provar a causa sem ID de pedido, mas tudo indica compra de duas unidades; remover subestimaria a demanda.`);

// --- A2. tendência e sazonalidade (v1: slide 8)
s = apendice("A2", "TENDÊNCIA E SAZONALIDADE", "Só dois padrões contam: degraus de nível e a semana de 2025");
legenda(s, `Máquina 1 · demanda média semanal: degraus × reta (treino até ${TD.fim.slice(8, 10)}/${TD.fim.slice(5, 7)}/${TD.fim.slice(0, 4)})`, M, 1.62, 6.4);
s.addChart([
  { type: pres.charts.LINE, data: [{ name: "Média semanal", labels: DG.rotulos, values: DG.demanda }, { name: "Nível do regime (degraus)", labels: DG.rotulos, values: DG.nivel }],
    options: { chartColors: [COR.contexto, COR.m1], lineSize: 2, lineDataSymbol: "none" } },
  { type: pres.charts.LINE, data: [{ name: "Tendência linear", labels: DG.rotulos, values: DG.reta }],
    options: { chartColors: [HEX.accent1], lineSize: 2, lineDash: "dash", lineDataSymbol: "none" } },
], eixos({ x: M, y: 1.88, w: 6.4, h: 2.4, valAxisMinVal: 0, valAxisMaxVal: 20, valAxisMajorUnit: 10, valAxisLabelFormatCode: "0",
           catAxisLabelFrequency: 13, showLegend: true, legendPos: "t", legendFontSize: 10, catAxisLabelFontSize: 10, valAxisLabelFontSize: 10 }));
// harmônicos da semana (2/7 e 3/7 ciclo/dia): as duas ordenadas da grade de Fourier (p/T) que cercam cada frequência
const T_PG = PG.periodo[0];
const vizinhos = [2 / 7, 3 / 7].map(h => [Math.floor(h * T_PG) - 1, Math.ceil(h * T_PG) - 1]);
const harmonicos = vizinhos.flat();
const rotulados = [0, 2, PG.pico, ...vizinhos.map(([a, b]) => PG.ordenada[a] >= PG.ordenada[b] ? a : b)];
const rotPG = PG.periodo.map((v, i) => rotulados.includes(i) ? `${fmt(v, v >= 10 ? 0 : 1)}` : "");
legenda(s, "Periodograma da M1 desde 01/02/2025 · azul: semana · tracejado: limite de 5%", M, 4.38, 6.4);
s.addChart([
  { type: pres.charts.BAR, data: [{ name: "Periodograma escalado", labels: rotPG, values: PG.ordenada }],
    options: { chartColors: PG.ordenada.map((_, i) => i === PG.pico ? COR.m1 : harmonicos.includes(i) ? "9DBFEA" : "CFC8C0"), barGapWidthPct: 35 } },
  { type: pres.charts.LINE, data: [{ name: "Limite de 5% (g_α)", labels: rotPG, values: PG.ordenada.map(() => PG.g_alfa) }],
    options: { chartColors: [HEX.accent1], lineSize: 1.5, lineDash: "dash", lineDataSymbol: "none" } },
], eixos({ x: M, y: 4.64, w: 6.4, h: 2.25, valAxisMinVal: 0, valAxisMaxVal: 0.4, valAxisMajorUnit: 0.2, valAxisLabelFormatCode: "0.0",
           showLegend: false, catAxisLabelFontSize: 10, valAxisLabelFontSize: 10, showCatAxisTitle: true, catAxisTitle: "período (dias)",
           catAxisTitleFontSize: 10, catAxisTitleColor: COR.eixo, catAxisTitleFontFace: "+mn-lt" }));
const achados = [
  ["Degraus, não tendência", `Os degraus explicam melhor que a reta (ΔBIC ${fmt(TD.dbic, 0)} na M1, ${fmt(TDT.dbic, 0)} no Total) e só ${TD.regimes_incl} de ${TD.regimes} regimes têm inclinação.`,
   "Planejar pelo patamar atual, sem projetar crescimento."],
  ["Período de 7 dias só em 2025, na M1", `Só com 2025, o teste de Fisher do teste.ipynb acha período 7 (p ${F25.m1_fev.p < 0.001 ? "< 0,001" : "= " + fmt(F25.m1_fev.p, 3)}); nos regimes de 2024, o dia da semana explica só ${pct0(SR.r2_antes_min)} a ${pct0(SR.r2_antes_max)}.`,
   "Abastecer por dia; padrão só das últimas 8 semanas."],
  ["Sem sazonalidade mensal ou anual", `No ano inteiro, os picos de ${TD.picos.replace(/ \/ /g, ", ")} dias eram efeito dos degraus; início do mês sem efeito (p = ${fmt(TD.p_dia_mes, 2)}).`,
   "Nenhum ajuste mensal; a anual exige 2 anos ou mais."],
  ["Sem ciclo, mas com inércia", `Nenhum período regular (p = ${fmt(TD.p_ciclo, 2)}), mas semanas boas e ruins vêm em sequência (p = ${fmt(TD.p_lb, 2)}).`,
   "Modelos de nível adaptativo: Holt‑Winters sem tendência."],
];
achados.forEach((a, i) => {
  const x = 7.25, y = 1.7 + i * 1.29, w = W - M - 7.25, h = 1.17;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.1, fill: { color: C.background2 }, line: { type: "none" }, objectName: `achado_${i}` });
  circulo(s, x + 0.18, y + 0.16, 0.42, C.accent1, String(i + 1));
  s.addText([{ text: a[0], options: { bold: true, fontSize: 15, color: C.text2, fontFace: "Cambria", breakLine: true } },
             { text: a[1] + " ", options: { fontSize: 12, color: C.text1 } },
             { text: "→ " + a[2], options: { fontSize: 12, bold: true, color: C.accent1 } }],
    { x: x + 0.75, y: y + 0.1, w: w - 0.9, h: h - 0.2, fontFace: "Calibri", valign: "top", margin: 0, isTextBox: true });
});
s.addNotes(`Resumo do notebook 05, só com o que muda uma decisão. Diagnósticos de tendência, ciclo e mês com o treino até 09/03/2025; o padrão semanal por regime usa a série completa (detalhe no slide 7). (1) Na série inteira, há uma 'tendência' de ~${fmt(TD.incl_ano, 1)}% por semana, mas ela é a soma de degraus: dentro dos regimes some, e a única exceção é uma queda no regime de mai–jul/2024. (2) A semana só aparece com a clientela de 2025. A mesma metodologia do teste.ipynb (periodograma + g de Fisher), aplicada só aos dados desde 01/02/2025, acha período fundamental 7 na M1 (pico em ${fmt(F25.m1_fev.pico, 1)} dias, g = ${fmt(F25.m1_fev.g, 2)} contra g_α = ${fmt(F25.m1_fev.g_alfa, 2)}) e no Total (p = ${fmt(F25.total_fev.p, 3)}). O resultado depende de ficar dentro do regime: desde a quebra de 30/01, p = ${fmt(F25.m1_quebra.p, 2)}; incluindo janeiro, p = ${fmt(F25.m1_jan.p, 2)} (limítrofe, janeiro é do regime antigo); só março (23 dias) não tem poder (p = ${fmt(F25.m1_mar.p, 2)}); só com o treino até 09/03, p = ${fmt(F25.m1_fev_treino.p, 2)}. A M2 não tem período 7. Em 2024, os perfis semanais mudam de forma a cada regime, como ruído. (3) O periodograma do ano inteiro (metodologia de teste.ipynb) acusa periodicidades de 62 a 187 dias, mas elas somem quando a tendência em degraus é retirada (p = ${fmt(TD.p_degraus, 2)}): eram o resíduo dos saltos. (4) O desvio em torno do nível é persistente, sem período: daí suavização exponencial / ARIMA(0,1,1), e nunca diferença sazonal (Δ7 é excessiva).`);

// --- A3. quebras estruturais (v1: slide 26)
s = apendice("A3", "QUEBRAS ESTRUTURAIS", "Quatro mudanças estruturais, todas com p < 0,001");
img(s, "quebras", M, 1.6, W - 2 * M, 4.0);
const marcos = [["14/05/2024", "janela de vendas 10–19h → 07–23h"], ["24/07/2024", "nível 7,0 → 11,4/dia"], ["16/11/2024", "nível 11,4 → 7,3/dia"], ["30/01/2025", "nível 7,3 → 14,4/dia"]];
marcos.forEach((m, i) => card(s, M + i * 3.07, 5.75, 2.9, 1.15, m[0], m[1], { tamanho: 20, rotulo: 12 }));
s.addNotes("Sup-Wald de Andrews com erros HAC, porque escolher a data olhando os dados invalida o Chow comum. As quebras são de nível, não de tendência. A de 24/07 vem logo depois da redução de preços de julho e a de 16/11, depois da alta de outubro: é compatível com elasticidade-preço, mas não é um teste causal.");

// --- A4. modelos avaliados (v1: slide 25)
s = apendice("A4", "MODELOS AVALIADOS", "Seis modelos complementares, sem redundância");
const cab = (t) => ({ text: t, options: { bold: true, color: HEX.lt1, fill: { color: HEX.dk2 }, fontSize: 13 } });
const linhaT = (a, b, c) => [{ text: a, options: { bold: true, fontSize: 13 } }, { text: b, options: { fontSize: 12 } }, { text: c, options: { fontSize: 12 } }];
s.addTable([
  [cab("Modelo"), cab("Por que entra"), cab("Limitação")],
  linhaT("Naive sazonal (baseline)", "referência mínima: repete a última semana", "carrega o ruído da última semana"),
  linhaT("Média 4 semanas (baseline)", "estável; o adversário mais exigente", "atrasa nas mudanças de nível"),
  linhaT("Holt-Winters (ETS)", "nível que se adapta rápido + sazonalidade semanal", "não usa variáveis explicativas"),
  linhaT("SARIMAX", "autocorrelação + dummies de quebra e máquinas ativas", "instável logo depois das quebras"),
  linhaT("Poisson / NB2", "contagens: variância ∝ média, intervalos coerentes", "supõe que o regime vigente persiste"),
  linhaT("Random Forest", "não linearidades; regra de 1 erro-padrão", "não extrapola; precisa de histórico"),
], { x: M, y: 1.65, w: 8.4, colW: [2.3, 3.3, 2.8], fontFace: "Calibri", color: HEX.dk1, border: { type: "solid", pt: 0.5, color: "D9D6D2" }, rowH: 0.62, valign: "middle" });
caixaTexto(s, "Testados antes e descartados", ["Croston/SBA: a demanda não é intermitente", "GARCH: heterocedasticidade é de nível", "Prophet, TBATS, Theta: inadequados ou redundantes", "XGBoost/LightGBM: piores e menos estáveis em 11 origens", "NHITS/NBEATSx: memorizam o ruído"], 9.25, 1.65, 3.5, 4.4, { tam: 13, tituloTam: 15 });
s.addNotes("A seleção partiu do notebook modelagens_testes, que comparou 16 modelos e duas redes neurais em 11 origens. Cada descarte tem motivo metodológico e evidência numérica (docs/methodology.md).");

// --- A5. engenharia de atributos (v1: slide 14)
s = apendice("A5", "ENGENHARIA DE ATRIBUTOS", "Variáveis usadas: calendário, histórico recente e quebras");
legenda(s, `${AT.n["Máquina 1"]} atributos na M1, ${AT.n["Total"]} no Total e ${AT.n["Máquina 2"]} na M2 (sem dados de cartão) · todos calculados só com dados até a véspera`, M, 1.62, W - 2 * M);
const cabA = (t) => ({ text: t, options: { bold: true, color: HEX.lt1, fill: { color: HEX.dk2 }, fontSize: 12 } });
const linA = (a, b, c, d) => [{ text: a, options: { bold: true, fontSize: 12 } }, { text: b, options: { fontSize: 11 } }, { text: c, options: { fontSize: 11 } }, { text: d, options: { fontSize: 11 } }];
s.addTable([
  [cabA("Grupo"), cabA("Variáveis"), cabA("Por que entra"), cabA("Usado em")],
  linA("Calendário (12)", "dia da semana, fim de semana, dia do mês, mês, semana do ano, feriado, data comercial, início do mês, Fourier semanal (4)", "padrão semanal e datas especiais", "Random Forest, Poisson/NB2; no SARIMAX, feriado, data comercial e início do mês"),
  linA("Defasagens (7)", "vendas de 1, 2, 3, 7, 14, 21 e 28 dias atrás", "inércia e repetição semanal", "Random Forest"),
  linA("Janelas móveis (9)", "média, desvio-padrão e CV de 7, 14 e 30 dias", "nível recente da demanda", `Random Forest: as mais importantes (sem elas, o erro sobe ${fmt(AT.jan_min, 0)}% a ${fmt(AT.jan_max, 0)}%)`),
  linA("Quebras (4–5)", "horário reduzido (antes de 18/05/2024), novos clientes (desde 30/01/2025), regimes intermediários, máquinas ativas (Total)", `os degraus de nível: ${fmt(AT.irr_horario, 2)}× e ${fmt(AT.irr_novos, 2)}× a demanda da M1`, "SARIMAX, Poisson/NB2, Random Forest"),
  linA("Comportamentais (4–6)", "% de vendas de manhã e à noite, ticket médio, itens por pedido; na M1, cartões únicos e % de cartões novos", "mudança de público", "Random Forest, só se a validação aprovar (não aprovou na M2)"),
  linA("Tendência (1)", "dias desde 01/03/2024", "controle", "Random Forest (sem efeito)"),
], { x: M, y: 1.95, w: W - 2 * M, colW: [1.95, 4.6, 2.45, 3.13], fontFace: "Calibri", color: HEX.dk1, border: { type: "solid", pt: 0.5, color: "D9D6D2" }, rowH: 0.52, valign: "middle", margin: [2, 5, 2, 5] });
[["Holt-Winters (M1)", "nenhuma variável externa: só nível e semana das últimas 8 semanas"],
 ["SARIMAX (Total)", `exógenas escolhidas por AICc: ${AT.exog_total.split(", ").map(v => ({ maquinas_ativas: "máquinas ativas", data_comercial: "data comercial", inicio_mes: "início do mês", dummy_novos_clientes: "novos clientes", feriado: "feriado" }[v] || v)).join(", ")}`],
 ["Sem vazamento", "dados só até a véspera; 2 testes automáticos no CI (futuro trocado por ruído não muda nada)"]]
  .forEach((c, i) => {
    const x = M + i * 4.1, y = 5.9, w = 3.85, h = 0.98;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.1, fill: { color: C.background2 }, line: { type: "none" }, objectName: `atr_${i}` });
    s.addText([{ text: c[0], options: { bold: true, fontSize: 13, color: C.text2, breakLine: true } }, { text: c[1], options: { fontSize: 12, color: C.text1 } }],
      { x: x + 0.18, y: y + 0.08, w: w - 0.36, h: h - 0.16, fontFace: "Calibri", valign: "top", margin: 0, isTextBox: true });
  });
s.addNotes(`Cada modelo usa um subconjunto. O Holt-Winters da M1 não usa variáveis externas: o nível e o padrão semanal são reestimados nas últimas 8 semanas. O SARIMAX escolhe as exógenas por AICc. A Random Forest recebe todas, mas a importância por permutação mostra que só as janelas móveis contam (os outros grupos mudam o erro em no máximo ${fmt(AT.outros_max, 1)}%), porque elas já carregam o nível vigente. Nos modelos estatísticos, as dummies de quebra são a forma de representar o novo patamar: com as duas dummies, o AIC cai ${fmt(Math.abs(AT.daic_m1), 0)} (M1) e ${fmt(Math.abs(AT.daic_total), 0)} (Total). Os atributos comportamentais são congelados na origem, porque não dá para prevê-los.`);

// --- A6. importância das variáveis (v1: slide 17)
const outrosIM = [...IM.m1.slice(1), ...IM.total.slice(1)];
s = apendice("A6", "IMPORTÂNCIA DAS VARIÁVEIS", "Só as médias móveis importam: elas já carregam o patamar");
legenda(s, "Random Forest · quanto o erro (MAE) sobe ao embaralhar cada grupo, fora da amostra · demanda", M, 1.62, 7.7);
s.addChart(pres.charts.BAR, [
  { name: "Máquina 1", labels: IM.grupos, values: IM.m1.map(v => Math.round(v) / 100) },
  { name: "Total", labels: IM.grupos, values: IM.total.map(v => Math.round(v) / 100) },
], eixos({ x: M, y: 1.95, w: 7.7, h: 4.9, barDir: "bar", barGrouping: "clustered", barGapWidthPct: 45, chartColors: [COR.m1, COR.total],
          showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "+0%;-0%;0%", catAxisOrientation: "maxMin",
          valAxisMinVal: -0.1, valAxisMaxVal: 0.4, valAxisMajorUnit: 0.1, valAxisLabelFormatCode: "+0%;-0%;0%",
          showLegend: true, legendPos: "t", catAxisLabelFontSize: 12, catAxisLabelPos: "low", dataLabelFontSize: 10 }));
[[`+${fmt(IM.m1[0], 0)}% / +${fmt(IM.total[0], 0)}%`, "de erro na M1 / no Total sem as médias móveis de 7, 14 e 30 dias"],
 [`${sinal(Math.min(...outrosIM))} a ${sinal(Math.max(...outrosIM))}`, "nos demais grupos: calendário, defasagens, quebras e comportamentais não somam informação"],
 ["redundância", "as médias recentes já refletem o patamar: dummies e calendário ficam redundantes na RF"],
 [`−${fmt(Math.abs(AT.daic_m1), 0)} de AIC`, "nos modelos estatísticos, ao contrário, as dummies de quebra são essenciais (Poisson/NB2 da M1)"]]
  .forEach((c, i) => card(s, 8.6, 1.7 + i * 1.31, W - M - 8.6, 1.18, c[0], c[1], { tamanho: 22, rotulo: 12 }));
s.addNotes(`Importância por permutação: o grupo inteiro é embaralhado nas 6 semanas antes da origem e mede-se quanto o MAE sobe. É fora da amostra, então não premia sobreajuste. Vale para a Random Forest treinada na M1 e no Total (candidata nas duas). O Holt-Winters escolhido para a M1 não usa variáveis externas, e a M2 não tem histórico para essa análise. Na receita, o padrão se repete: +${fmt(IM.receita_jan["Máquina 1"], 0)}% (M1) e +${fmt(IM.receita_jan["Total"], 0)}% (Total) sem as médias móveis. Valores negativos pequenos são ruído: embaralhar o grupo não piora o erro.`);

// --- A7. evidência por máquina (v1: slide 19)
s = apendice("A7", "EVIDÊNCIA POR MÁQUINA", "Os modelos escolhidos superam o naive nas duas máquinas");
const graficoEvid = (x, titulo, series, cores) => {
  legenda(s, titulo, x, 1.62, 5.95);
  s.addChart([
    { type: pres.charts.BAR, data: series.map(([nome, v]) => ({ name: nome, labels: EV.modelos, values: v })),
      options: { chartColors: cores, barGrouping: "clustered", barGapWidthPct: 45, showValue: true } },
    { type: pres.charts.LINE, data: [{ name: "Naive = 1", labels: EV.modelos, values: EV.modelos.map(() => 1) }],
      options: { chartColors: [HEX.accent1], lineSize: 1.5, lineDash: "dash", lineDataSymbol: "none", showValue: false } },
  ], eixos({ x, y: 1.92, w: 5.95, h: 3.45, valAxisMinVal: 0, valAxisMaxVal: 1.2, valAxisMajorUnit: 0.2, valAxisLabelFormatCode: "0.0",
             dataLabelFormatCode: "0.00", dataLabelFontSize: 9, showLegend: true, legendPos: "t", legendFontSize: 11,
             catAxisLabelFontSize: 10, valAxisLabelFontSize: 10 }));
};
graficoEvid(M, "Máquina 1 · backtest de 11 janelas de 14 dias · MASE médio (1 = naive)", [["Demanda", EV.m1_dem], ["Receita", EV.m1_rec]], [COR.m1, "9DBFEA"]);
graficoEvid(M + 6.18, "Máquina 2 · demanda · erro (MAE) ÷ erro do naive · sem backtest (30 dias de treino)", [["Validação interna", EV.m2_val], ["Teste", EV.m2_teste]], [COR.m2, "F5B590"]);
const iRF = EV.modelos.indexOf("Random Forest");
[["Máquina 1 · 11 janelas", `Holt-Winters: MASE ${fmt(m1d.mase_bt, 2)}, o melhor na demanda; ~${bt_ganho_m1}% abaixo do naive, mas não significativo após Holm (p = ${fmt(m1d.p_holm_naive, 2)})`],
 ["Máquina 2 · validação e teste", `Random Forest: o menor erro nas duas etapas (${fmt(EV.m2_val[iRF], 2)}× e ${fmt(EV.m2_teste[iRF], 2)}× o naive); na receita, ${fmt(EV.m2_teste_rec[iRF], 2)}× no teste`],
 ["Diferenças pequenas", `na M2, nenhum modelo difere do naive com significância (Diebold-Mariano p ≥ ${fmt(D.dm_m2_p_min, 2)}); no Total, ${EV.total_signif} candidatos superam o naive (p Holm ≤ ${fmt(EV.total_p_max, 3)})`]]
  .forEach((c, i) => {
    const x = M + i * 4.1, y = 5.55, w = 3.85, h = 1.32;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.1, fill: { color: C.background2 }, line: { type: "none" }, objectName: `ev_${i}` });
    s.addText([{ text: c[0], options: { bold: true, fontSize: 13, color: C.text2, breakLine: true } }, { text: c[1], options: { fontSize: 12, color: C.text1 } }],
      { x: x + 0.18, y: y + 0.08, w: w - 0.36, h: h - 0.16, fontFace: "Calibri", valign: "top", margin: 0, isTextBox: true });
  });
s.addNotes(`Máquina 1: MASE médio em 11 blocos de 14 dias sem sobreposição, com tudo refeito em cada origem (MASE < 1 = erro menor que o naive sazonal). Máquina 2: com só 30 dias de treino, não há backtest; a escolha usou a validação interna (3 dobras de 14 dias dentro do treino) e o teste. As barras mostram o MAE de cada modelo dividido pelo MAE do naive na mesma etapa. Na validação, todos os modelos batem o naive; no teste, Média 4 semanas e Holt-Winters ficam piores que ele, e a Random Forest é a melhor nas duas etapas. Com 14 dias e ~6 vendas/dia, as diferenças na M2 não são significativas, por isso a previsão dela é tratada como nível médio com faixa larga.`);

// --- A8. métricas de qualidade no teste (v1: slide 24)
s = apendice("A8", "MÉTRICAS DE QUALIDADE", "Qualidade no teste: erro de ~20% na M1 e no Total, ~40% na M2");
legenda(s, "Teste de 10 a 23/03/2025 (14 dias) · modelos escolhidos; no Total, a soma recomendada e o modelo único como referência", M, 1.62, W - 2 * M);
const cabQ = (t) => ({ text: t, options: { bold: true, color: HEX.lt1, fill: { color: HEX.dk2 }, fontSize: 12, align: "center" } });
const linhasQ = [[cabQ("Série · alvo"), cabQ("Modelo"), cabQ("MAE"), cabQ("RMSE"), cabQ("MAPE"), cabQ("WAPE"), cabQ("R²"), cabQ("Viés"),
                  cabQ("Cobertura 80%"), cabQ("Cobertura 95%"), cabQ("MASE backtest")]];
[["Máquina 1", "demanda"], ["Máquina 1", "receita"], ["Máquina 2", "demanda"], ["Máquina 2", "receita"], ["Total", "demanda", "Soma M1 + M2"],
 ["Total", "receita", "Soma M1 + M2"], ["Total", "demanda", "SARIMAX"], ["Total", "receita", "SARIMAX"]]
  .forEach(([se, al, mo]) => {
    const r = ql(se, al, mo), dd = al === "demanda" ? 1 : 0, ref = mo === "SARIMAX";
    const o = (bold) => ({ fontSize: 12, align: "center", color: ref ? HEX.accent6 : HEX.dk1, bold: !!bold, italic: ref });
    const num = (v, d) => (v === null || v === undefined) ? "—" : fmt(v, d);
    const pc = (v) => (v === null || v === undefined) ? "—" : `${fmt(v, 0)}%`;
    linhasQ.push([
      { text: `${se} · ${al}`, options: { ...o(true), align: "left" } },
      { text: ref ? "SARIMAX (referência)" : r.modelo, options: { ...o(!ref), align: "left" } },
      { text: num(r.MAE, dd), options: o() }, { text: num(r.RMSE, dd), options: o() },
      { text: pc(r["MAPE_%"]), options: o() }, { text: pc(r["WAPE_%"]), options: o(!ref) },
      { text: num(r.R2, 2), options: o() }, { text: sinal(r["ME_relativo_%"], 0), options: o() },
      { text: pc(r["cobertura_80_%"]), options: o() }, { text: pc(r["cobertura_95_%"]), options: o() },
      { text: num(r["MASE backtest"], 2), options: o() }]);
  });
s.addTable(linhasQ, { x: M, y: 1.95, w: W - 2 * M, colW: [1.95, 2.1, 0.85, 0.85, 0.8, 0.8, 0.7, 0.8, 1.13, 1.13, 1.02], fontFace: "Calibri",
  color: HEX.dk1, border: { type: "solid", pt: 0.5, color: "D9D6D2" }, rowH: 0.4, valign: "middle", margin: [1, 4, 1, 4] });
[["Como ler", "MAE e RMSE em vendas/dia ou u.m./dia; WAPE = erro total ÷ volume; R² = quanto da variação diária a previsão acompanha"],
 ["R² baixo na M2", `a previsão acerta o nível, não o dia a dia: com ~6 vendas/dia, o ruído de contagem (piso de WAPE ≈ ${fmt(D.piso_ruido["Máquina 2"], 0)}%) domina`],
 ["Soma × modelo único", `a soma erra menos em média (WAPE ${fmt(sd["WAPE_%"], 0)}% × ${fmt(sx["WAPE_%"], 0)}%), mas tem RMSE maior na demanda (${fmt(sd.RMSE, 1)} × ${fmt(sx.RMSE, 1)}): alguns dias com erro grande`]]
  .forEach((c, i) => {
    const x = M + i * 4.1, y = 5.75, w = 3.85, h = 1.12;
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.1, fill: { color: C.background2 }, line: { type: "none" }, objectName: `ql_${i}` });
    s.addText([{ text: c[0], options: { bold: true, fontSize: 13, color: C.text2, breakLine: true } }, { text: c[1], options: { fontSize: 12, color: C.text1 } }],
      { x: x + 0.18, y: y + 0.08, w: w - 0.36, h: h - 0.16, fontFace: "Calibri", valign: "top", margin: 0, isTextBox: true });
  });
s.addNotes(`Métricas no teste (14 dias), que entra na seleção; o desempenho esperado vem do MASE do backtest (11 janelas). MAPE não existe na M2 porque há dias sem venda. Viés = erro médio ÷ volume (positivo = superestima). Os intervalos de 80% cobriram 71% a 100% dos dias e os de 95%, todos: estão calibrados ou um pouco largos. Nenhum modelo deixou autocorrelação nos erros de teste (Ljung-Box p > 0,05). Ressalva sobre a recomendação: na demanda do Total, a soma tem MAE e WAPE menores, mas RMSE maior e R² menor (${fmt(sd.R2, 2)} × ${fmt(sx.R2, 2)}) que o SARIMAX: acerta mais dias, mas erra alguns por mais. Na receita, a soma é melhor em todas as métricas. É uma janela de 14 dias; o monitoramento das duas abordagens continua.`);

// --- A9. arquitetura para produção (v1: slide 27)
s = apendice("A9", "ARQUITETURA PARA PRODUÇÃO", "Pipeline batch diário, versionado e monitorado");
const etapas = ["Logs das máquinas", "Validação de contrato", "Séries diárias", "Quebras + atributos", "Treino e seleção", "Registro de modelos", "Previsão de 14 dias"];
etapas.forEach((e, i) => {
  const x = M + i * 1.75;
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 2.0, w: 1.55, h: 1.15, rectRadius: 0.1, fill: { color: i === 6 ? C.accent1 : C.text2 }, line: { type: "none" }, objectName: `etapa_${i}` });
  s.addText(e, { x, y: 2.0, w: 1.55, h: 1.15, align: "center", valign: "middle", fontFace: "Calibri", fontSize: 13, bold: true, color: C.background1, margin: 3, isTextBox: true });
  if (i < etapas.length - 1) s.addShape(pres.shapes.LINE, { x: x + 1.57, y: 2.575, w: 0.16, h: 0, line: { color: HEX.accent6, width: 2, endArrowType: "triangle" }, objectName: `seta_${i}` });
});
caixaTexto(s, "Monitoramento (diário)", ["WAPE e viés móveis de 28 dias", "soma das máquinas × modelo único do Total", "cobertura do intervalo de 80%", "dia fora do IC 95% = anomalia"], M, 3.6, 3.9, 2.9, { tam: 13, tituloTam: 15 });
caixaTexto(s, "Drift", ["dados: PSI/KS da demanda, ticket, mix e horário", "conceito: detecção online de quebras + CUSUM dos resíduos", "esquema: contrato bloqueia a carga"], M + 4.1, 3.6, 4.0, 2.9, { tam: 13, tituloTam: 15 });
caixaTexto(s, "Retreino e versão", ["semanal; seleção completa mensal", "gatilho: alerta por 7 dias ou nova quebra", "modelo + hash de dados e configuração", "shadow mode antes de promover"], M + 8.3, 3.6, 3.85, 2.9, { tam: 13, tituloTam: 15 });
s.addNotes("Batch, e não streaming: dezenas de vendas por dia e decisões diárias. O pipeline completo roda em ~10 minutos. O drift de mix e ticket é monitorado porque foi exatamente isso que mudou em 2025.");

// --- A10. ampliação do horário: inferência da janela e efeito (os testes que saíram do slide 6)
{
  s = apendice("A10", "AMPLIAÇÃO DO HORÁRIO", `Em ${dataBR(HR.data)} a M1 ampliou o horário: 10–19h → 07–23h`);
  img(s, "janela", M, 1.7, 8.35, 4.0, "esquerda");
  s.addText(`O horário não está na base: a janela é inferida pela primeira e pela última venda de cada dia. A data foi detectada por Binseg + BIC na proporção diária de vendas fora de 10–19h, só com o treino (${dataBR(HR.data_treino)}), e refinada com a série inteira (${dataBR(HR.data)}, o primeiro dia com vendas fora da janela); Sup-Wald de Andrews com p < 0,001. Efeito em ±8 semanas: demanda total IRR ${fmt(HR.irr_total, 2)} (p = ${fmt(HR.p_total, 3)}); núcleo 10–19h IRR ${fmt(HR.irr_nucleo, 2)} (p = ${fmt(HR.p_nucleo, 2)}).`,
    { x: M, y: 5.75, w: 8.35, h: 1.15, fontFace: "Calibri", fontSize: 13, color: C.text1, valign: "top", margin: 0, isTextBox: true });
  [[`${fmt(HR.fora_antes, 0)}% → ${fmt(HR.fora_depois, 0)}%`, "das vendas da M1 fora do horário original (10–19h), antes × depois"],
   ["07–23h", "nova janela de vendas, inferida pela primeira e última venda do dia (antes, ~10–19h)"],
   [sinal(100 * (HR.irr_total - 1)), `de demanda total nas 8 semanas seguintes (p = ${fmt(HR.p_total, 3)}); o núcleo 10–19h não caiu`],
   [`${fmt(TP["M1 · base anterior (mesma tabela)"].noite, 0)}% → ${fmt(TP["M1 · desde a quebra (todos)"].noite, 0)}%`, "das vendas após 19h com o público de 2025: a noite perdeu peso"]]
    .forEach((c, i) => card(s, 9.2, 1.7 + i * 1.31, W - M - 9.2, 1.18, c[0], c[1], { tamanho: 22, rotulo: 12 }));
  s.addNotes(`Detalhe do slide 6. A base não registra o horário de funcionamento. A janela foi inferida pela primeira e última venda do dia, e a data foi detectada por Binseg + BIC na proporção diária de vendas fora de 10–19h, só com o treino (${dataBR(HR.data_treino)}), e refinada com a série inteira (${dataBR(HR.data)}, o primeiro dia com vendas fora da janela). Sup-Wald de Andrews com p < 0,001. Efeito em ±8 semanas: demanda total IRR ${fmt(HR.irr_total, 2)} (p = ${fmt(HR.p_total, 3)}); núcleo 10–19h IRR ${fmt(HR.irr_nucleo, 2)} (p = ${fmt(HR.p_nucleo, 2)}). A ampliação explica o primeiro degrau da série (mai/2024); o efeito sobre as vendas do horário original é testado como canibalização (slide 9 e A11). Implicação de negócio: em 2025, com o novo público, só 9% das vendas acontecem depois das 19h; vale avaliar o custo de manter a máquina abastecida e ligada até 23h.`);
}

// --- A11. testes de canibalização e coortes (versão completa do slide 9)
{
s = apendice("A11", "CANIBALIZAÇÃO E NOVOS CLIENTES", "Testes de canibalização e coortes de cartões");
const bandas = [
  { nome: "(a) Horário ampliado", quando: "M1 · faixas horárias · mai/2024", veredito: "Canibalização parcial no horário", imagem: "canib_horario", y: 1.65, h: 1.5,
    evid: [[`IRR ${fmt(HR.irr_nucleo, 2)}`, ` (p = ${fmt(HR.p_nucleo, 2)}): núcleo 10–19h estável`],
           [sinal(100 * (HR.irr_total - 1)), ` de demanda total (p = ${fmt(HR.p_total, 3)})`],
           ["incrementalidade 0,84", " (IC 0,34–1,23): canibalização parcial não excluída"]] },
  { nome: "(b) Entrada da M2", quando: "M1 antes × depois de 08/02/2025", veredito: "Sem perda líquida com a M2", imagem: "canib_m2", y: 3.27, h: 1.5,
    evid: [["14,2 → 14,4/dia", " na M1 (IRR 1,06): sem queda líquida"],
           ["ρ = −0,11", ": sem substituição diária entre as máquinas"],
           ["~0,6/dia", " a menos na M1 à noite: indício localizado, confundido com a troca de público"]] },
  { nome: "(c) Salto de 2025", quando: "M1 · cartões novos × antigos", veredito: "Nova clientela na Máquina 1", imagem: "clientes", y: 4.89, h: 2.01,
    evid: [["177 de 196", " cartões ativos após 30/01/2025 nunca tinham comprado"],
           ["13,0 de 14,4", " vendas/dia após a quebra vêm de cartões novos"],
           ["7,0 → 1,4", " vendas/dia dos cartões antigos: a base anterior quase sumiu"],
           ["3,8 × 1,5", " compras por cartão em 6 semanas (nova coorte × anteriores)"]] },
];
const xEv = 7.37, wEv = W - M - xEv;
bandas.forEach((b, k) => {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: M, y: b.y, w: 2.5, h: b.h, rectRadius: 0.1, fill: { color: C.background2 }, line: { type: "none" }, objectName: `banda_rotulo_${k}` });
  s.addText([{ text: b.nome, options: { bold: true, fontSize: 15, color: C.text2, fontFace: "Cambria", breakLine: true } },
             { text: b.quando, options: { fontSize: 11, color: C.accent6 } }],
    { x: M + 0.18, y: b.y + 0.12, w: 2.14, h: 0.8, fontFace: "Calibri", valign: "top", margin: 0, isTextBox: true });
  s.addText(`→ ${b.veredito}`, { x: M + 0.18, y: b.y + b.h - 0.62, w: 2.14, h: 0.5, fontFace: "Calibri", fontSize: 13, bold: true, color: C.accent1, valign: "bottom", margin: 0, isTextBox: true });
  img(s, b.imagem, 3.22, b.y + 0.04, 4.0, b.h - 0.08);
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: xEv, y: b.y, w: wEv, h: b.h, rectRadius: 0.1, fill: { color: C.background2 }, line: { type: "none" }, objectName: `banda_evid_${k}` });
  const runs = [];
  b.evid.forEach(([num, resto], i) => {
    runs.push({ text: num, options: { bold: true, color: C.accent1, bullet: true, paraSpaceAfter: 4 } });
    runs.push({ text: resto, options: { color: C.text1, breakLine: i < b.evid.length - 1 } });
  });
  s.addText(runs, { x: xEv + 0.15, y: b.y + 0.1, w: wEv - 0.3, h: b.h - 0.2, fontFace: "Calibri", fontSize: 12, valign: "middle", margin: 2, isTextBox: true });
});
s.addNotes("Detalhe do slide 9, com todos os testes. Pergunta: a expansão gerou demanda nova ou só redistribuiu vendas? Três testes, em ordem cronológica. O horário mudou em mai/2024 e a M2 só existe a partir de fev/2025, então a canibalização entre máquinas pela mudança de horário não é testável; por isso separamos (a) faixas horárias da M1 em 2024 e (b) efeito da entrada da M2 sobre a M1. (a) Com o horário ampliado, o total subiu 45% e o núcleo 10–19h ficou estável; a incrementalidade de 0,84 (IC 0,34–1,23) não exclui canibalização parcial. (b) Abrir a M2 somou volume, sem tirar vendas líquidas da M1; há só um indício localizado à noite, confundido com a troca de público. (c) O salto de 2025 veio de uma nova clientela: o proxy é o cartão (só a M1 tem). A nova clientela compra com mais frequência, mas prefere Americano, vende pouco à noite (9% das vendas depois das 19h, contra 21% antes) e quase nada no fim de semana. É compatível com mudança de local ou de contexto, mas a causa não é identificável: cartões podem ser reemitidos. Datas e testes das quebras (Sup-Wald de Andrews) em A3.");
}

// --- A12. soma das máquinas × modelo único do Total (a comparação que saiu do slide 11)
{
  s = apendice("A12", "SOMA × MODELO ÚNICO DO TOTAL", "Por que somar as máquinas em vez de um modelo único");
  legenda(s, "Teste de 10 a 23/03/2025 e próximos 14 dias: observado × modelo único do Total (SARIMAX) × soma M1 + M2 · demanda e receita", M, 1.6, W - 2 * M);
  img(s, "total_soma", M, 1.9, W - 2 * M, 3.0);
  const w4a = (W - 2 * M - 3 * 0.2) / 4, yb = 5.02, hb = 1.88;
  caixaTexto(s, "Demanda (erro médio no teste)", [`modelo único do Total: ${fmt(TS("demanda", "Total direto").MAE, 2)} vendas/dia`, `soma M1 + M2: ${fmt(TS("demanda", "Soma M1 + M2").MAE, 2)} (−${ganhoSoma("demanda")}%)`], M, yb, w4a, hb, { tam: 13, tituloTam: 13 });
  caixaTexto(s, "Receita (erro médio no teste)", [`modelo único do Total: ${fmt(TS("receita", "Total direto").MAE, 0)}${UM}/dia`, `soma M1 + M2: ${fmt(TS("receita", "Soma M1 + M2").MAE, 0)} (−${ganhoSoma("receita")}%)`], M + (w4a + 0.2), yb, w4a, hb, { tam: 13, tituloTam: 13 });
  caixaTexto(s, "Por que a soma faz sentido", ["M1 vende em dias úteis e M2 no fim de semana: o modelo único achata os dois padrões", "reposição e rota são por máquina"], M + 2 * (w4a + 0.2), yb, w4a, hb, { tam: 12, tituloTam: 13, espaco: 3 });
  caixaTexto(s, "Ressalvas", ["1 janela de 14 dias, sem backtest", `demanda: RMSE ${fmt(sd.RMSE, 1)} × ${fmt(sx.RMSE, 1)}; R² ${fmt(sd.R2, 2)} × ${fmt(sx.R2, 2)}`, `SARIMAX em paralelo: ${fmt(ft.demanda.media)} vendas/dia`], M + 3 * (w4a + 0.2), yb, w4a, hb, { tam: 12, tituloTam: 13, espaco: 3 });
  s.addNotes(`Detalhe do slide 11. No gráfico, a soma acompanha os vales de fim de semana (por exemplo, 16/03) que o modelo único suaviza; no futuro, o modelo único chega a ficar plano. Na demanda, a soma tem MAE e WAPE menores (WAPE ${fmt(sd["WAPE_%"], 0)}% × ${fmt(sx["WAPE_%"], 0)}%), mas RMSE maior e R² menor que o SARIMAX: acerta mais dias, mas erra alguns por mais; na receita, a soma é melhor em todas as métricas (A8). Ressalva honesta: é uma janela de 14 dias e não há backtest possível, porque a M2 só tem 44 dias. O modelo único do Total segue em paralelo como alarme de coerência: divergência grande da soma = investigar. Próximo passo técnico: reconciliação hierárquica (MinT) quando houver histórico.`);
}

// ------------------------------------------------------------------ gravação + pós-processamento
// O pptxgenjs grava pontos nulos como <c:v></c:v>; removemos esses pontos para que a linha da M2 só
// comece em fev/2025 (ponto ausente = lacuna, como o Excel grava).
async function removerPontosVazios(arquivo) {
  const zip = await JSZip.loadAsync(fs.readFileSync(arquivo));
  for (const nome of Object.keys(zip.files).filter(n => /^ppt\/charts\/chart\d+\.xml$/.test(n))) {
    const xml = await zip.file(nome).async("string");
    const novo = xml.replace(/<c:pt idx="\d+"><c:v><\/c:v><\/c:pt>/g, "");
    if (novo !== xml) zip.file(nome, novo);
  }
  // regrava só arquivos (sem entradas de pasta, que alguns serviços de nuvem rejeitam)
  const limpo = new JSZip();
  for (const [nome, entrada] of Object.entries(zip.files)) {
    if (!entrada.dir) limpo.file(nome, await entrada.async("nodebuffer"), { createFolders: false });
  }
  fs.writeFileSync(arquivo, await limpo.generateAsync({ type: "nodebuffer", compression: "DEFLATE" }));
}

(async () => {
  await pres.writeFile({ fileName: SAIDA });
  await applyTheme(SAIDA, THEME);
  await removerPontosVazios(SAIDA);
  console.log("gerado:", SAIDA);
})();
