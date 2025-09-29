using Microsoft.AspNetCore.Builder;
using Microsoft.Extensions.Hosting;
using Microsoft.AspNetCore.Http;

var builder = WebApplication.CreateBuilder(args);

var app = builder.Build();

app.UseDefaultFiles();
app.UseStaticFiles();

app.MapPost("/api/evaluate", async (HttpContext context) => {

    var requestBody = await new StreamReader(context.Request.Body).ReadToEndAsync();

    string[] arguments = {"40", "1.0"};

    string response = Driver.WebRun(arguments, requestBody);

    await context.Response.WriteAsync(response);
});

app.Run();
