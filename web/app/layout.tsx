export const metadata = {
  title: "SentiQuant 논조 분석 데모",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="ko">
      <body style={{ fontFamily: "sans-serif", maxWidth: 720, margin: "40px auto", padding: "0 16px" }}>
        {children}
      </body>
    </html>
  );
}
