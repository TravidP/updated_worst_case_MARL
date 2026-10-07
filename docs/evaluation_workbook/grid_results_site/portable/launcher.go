// Standalone local server for the CBWCE results viewer. Standard library only.
package main

import (
	"context"
	"flag"
	"fmt"
	"net"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"runtime"
	"time"
)

func siteDirectory() (string, error) {
	binary, err := os.Executable()
	if err != nil {
		return "", err
	}
	directory := filepath.Dir(binary)
	for i := 0; i < 4; i++ {
		candidate := filepath.Join(directory, "dist")
		if info, err := os.Stat(filepath.Join(candidate, "index.html")); err == nil && !info.IsDir() {
			return candidate, nil
		}
		directory = filepath.Dir(directory)
	}
	return "", fmt.Errorf("dist/index.html is missing. Extract the entire ZIP before starting")
}

func openBrowser(url string) error {
	var command *exec.Cmd
	switch runtime.GOOS {
	case "windows":
		command = exec.Command("rundll32.exe", "url.dll,FileProtocolHandler", url)
	case "darwin":
		command = exec.Command("open", url)
	default:
		command = exec.Command("xdg-open", url)
	}
	if err := command.Start(); err != nil {
		return err
	}
	go command.Wait()
	return nil
}

func run() error {
	port := flag.Int("port", 8878, "Local port; 0 chooses a free port")
	noBrowser := flag.Bool("no-browser", false, "Do not open the browser automatically")
	flag.Parse()
	if *port < 0 || *port > 65535 {
		return fmt.Errorf("port must be between 0 and 65535")
	}
	directory, err := siteDirectory()
	if err != nil {
		return err
	}
	listener, err := net.Listen("tcp", fmt.Sprintf("127.0.0.1:%d", *port))
	if err != nil && *port == 8878 {
		fmt.Println("Port 8878 is busy; choosing another available local port.")
		listener, err = net.Listen("tcp", "127.0.0.1:0")
	}
	if err != nil {
		return err
	}
	url := "http://" + listener.Addr().String() + "/"
	fmt.Printf("CBWCE Results - Protocol v7\n\nOpen: %s\nServing: %s\n\nKeep this window open. Press Ctrl+C or close this window to stop.\n", url, directory)
	files := http.FileServer(http.Dir(directory))
	server := &http.Server{ReadHeaderTimeout: 10 * time.Second, Handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet && r.Method != http.MethodHead {
			w.Header().Set("Allow", "GET, HEAD")
			http.Error(w, "Read-only viewer", http.StatusMethodNotAllowed)
			return
		}
		w.Header().Set("Cache-Control", "no-cache")
		files.ServeHTTP(w, r)
	})}
	stopped := make(chan os.Signal, 1)
	signal.Notify(stopped, os.Interrupt)
	defer signal.Stop(stopped)
	go func() {
		<-stopped
		ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
		defer cancel()
		server.Shutdown(ctx)
	}()
	if !*noBrowser {
		if err := openBrowser(url); err != nil {
			fmt.Printf("Open the URL above manually (%v).\n", err)
		}
	}
	if err := server.Serve(listener); err != nil && err != http.ErrServerClosed {
		return err
	}
	return nil
}

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, "Unable to start:", err)
		if runtime.GOOS == "windows" {
			fmt.Println("Press Enter to close.")
			var answer string
			fmt.Scanln(&answer)
		}
		os.Exit(1)
	}
}
