import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import Nav from "@/components/nav";

export default function LoginPage() {
  return (
    <div className="flex flex-col min-h-screen">
      <Nav />
      <main className="flex-1 flex items-center justify-center px-4">
        <Card className="w-full max-w-sm">
          <CardHeader className="text-center">
            <CardTitle>Sign in</CardTitle>
            <CardDescription>Authentication is coming in a future release.</CardDescription>
          </CardHeader>
          <CardContent className="flex flex-col gap-3">
            <Button disabled className="w-full">
              Continue with Google
            </Button>
            <Button variant="outline" disabled className="w-full">
              Continue with GitHub
            </Button>
            <p className="text-center text-xs text-muted-foreground pt-2">
              For now,{" "}
              <Link href="/docs" className="underline underline-offset-4 hover:text-foreground">
                explore the app without an account
              </Link>
              .
            </p>
          </CardContent>
        </Card>
      </main>
    </div>
  );
}
